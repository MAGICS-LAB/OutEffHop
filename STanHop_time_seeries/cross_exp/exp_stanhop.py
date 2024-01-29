from data.data_loader import Dataset_MTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.stanhop import STanHopNet
from torchmetrics.functional import r2_score

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import math
import os
import time
import json
import pickle

import warnings
# import math


warnings.filterwarnings('ignore')

class SimpleAverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


def attach_act_hooks(model):
    
    act_dict = OrderedDict()
    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
            
            act_dict[name] = (inp, out)
        return _hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def kurtosis(data):
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / n
    fourth_moment = sum((x - mean) ** 4 for x in data) / n
    return fourth_moment / (var ** 2)

def kurtosis_from_quantized_trandformer(x, eps=1e-6):
    """x - (B, d)"""

    # print("Shape of the input tensor of kurtosis", x.shape)
    mu = x.mean(dim=1, keepdims=True)
    # print("mean of input tensor of kurtosis", mu)
    s = x.std(dim=1)
    # print("standard deviation of input tensor of kurtosis", s)
    mu4 = ((x - mu) ** 4.0).mean(dim=1)
    k = mu4 / (s**4.0 + eps)
    # print("kurtosis:", k)
    return k

def hook_fn(module, input, output):
    with torch.no_grad(): module.output = output

class Exp_Stanhop(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stanhop, self).__init__(args)
    
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="STanHop-Net",
        #     name = self.args.run_name
        # )

    def _build_model(self):        
        model = STanHopNet(
            self.args.data_dim, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.e_layers,
            self.args.dropout, 
            self.args.baseline,
            self.device,
            self.args.eta,
            self.args.gamma,
            self.args.mode
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def __get_alpha(self):
        
        # return [0], [0], [0]
        time_alpha = []
        sec_alpha_1 = []
        sec_alpha_2 = []

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.time_attention.inner_attention.entmax.alpha.item()
                time_alpha.append(1+ (1/(1 + np.exp(-a))))

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.dim_sender.inner_attention.entmax.alpha.item()
                sec_alpha_1.append( 1+ (1/(1 + np.exp(-a))))

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.dim_receiver.inner_attention.entmax.alpha.item()
                sec_alpha_2.append(1+ (1/(1 + np.exp(-a))))

        return time_alpha, sec_alpha_1, sec_alpha_2

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
        )
        
        g = torch.Generator()
        g.manual_seed(0)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, global_step, log = False):
        
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        #wandb.log({'vali loss':total_loss})
        
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        log_data = {
            'epoch':[],
            'time attn alpha':[],
            'sec attn alpha 1': [],
            'sec attn alpha 2':[],
            'train loss':[]
        }
        global_step = 0
        
        act_dict = attach_act_hooks(self.model)
        
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                model_optim.step()
                global_step += 1
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, global_step, log = True)
            test_loss = self.vali(test_data, test_loader, criterion, global_step, log = False)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if math.isnan(vali_loss):
                raise RuntimeError("ERROR: loss is nan!")

            log_data = {
                'train loss': train_loss,
                'valid loss':vali_loss,
                'test loss':test_loss,
                'steps':train_steps,
                'epoch':epoch+1,
            }
            #wandb.log(log_data)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self.test(setting, False)
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')

        return self.model, early_stopping.val_loss_min

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        act_dict = attach_act_hooks(self.model)
        num_layers = len(self.model.decoder.decode_layers)
        ACT_KEYS = [
            *[f"model.decoder.decode_layers.{j}" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.MLP1" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.MLP2" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.norm2" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.norm1" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.hopfield.out_projection" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.sthm.norm4" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.sthm.norm3" for j in range(num_layers)],
            *[f"model.decoder.decode_layers.{j}.sthm.hopfield" for j in range(num_layers)],
        ]
        
        act_inf_norms = OrderedDict()
        act_kurtoses = OrderedDict()
        
        def hook_fn(module, input, output):
            with torch.no_grad(): module.output = output
            
        for b in self.model.decoder.decode_layers:
            b.MLP2.register_forward_hook(hook_fn);

            
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                for name in act_dict:
                    x_inp, x_out = act_dict[name]
                    x = x_out
                    if isinstance(x, list):
                        continue
                    x = x.detach().cpu().contiguous().view(x.size(0), -1)

                    # compute inf norm
                    inf_norms = x.norm(dim=1, p=np.inf)
                    if not name in act_inf_norms:
                        act_inf_norms[name] = SimpleAverageMeter()
                    for v in inf_norms:
                        act_inf_norms[name].update(v.item())

                    
                    kurt = kurtosis_from_quantized_trandformer(x)
                    if not name in act_kurtoses:
                        act_kurtoses[name] = SimpleAverageMeter()
                    for v in kurt:
                        act_kurtoses[name].update(v.item())
   
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        log_data = {
            'test mse':mse,
            'test mae':mae
        }
        print(log_data)
        #wandb.log(log_data)
        metrics = OrderedDict([("mae", mae), ("mse", mse)])
        
        for name, v in act_inf_norms.items():
            metrics[name] = v.avg
        max_inf_norm = [v.avg for v in act_inf_norms.values()]
        max_inf_norm = max(max_inf_norm)
        max_ffn_inf_norm = max(v.avg for k, v in act_inf_norms.items() if ".MLP" in k)
        max_layer_inf_norm = max(
            act_inf_norms[f"decoder.decode_layers.{j}"].avg for j in range(num_layers)
        )

        avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
        max_kurtosis = max(v.avg for v in act_kurtoses.values())
        max_kurtosis_layers = max(
            act_kurtoses[f"decoder.decode_layers.{j}"].avg for j in range(num_layers)
        )

        metrics["max_inf_norm"] = max_inf_norm
        metrics["max_ffn_inf_norm"] = max_ffn_inf_norm
        metrics["max_layer_inf_norm"] = max_layer_inf_norm
        metrics["avg_kurtosis"] = avg_kurtosis
        metrics["max_kurtosis"] = max_kurtosis
        metrics["max_kurtosis_layers"] = max_kurtosis_layers

        print(metrics)
        outputs = [b.MLP2.output for b in self.model.decoder.decode_layers]  
        print("Inf_norm_output:", max([o.abs().max() for o in outputs]))
        print("kurtosis_output:", kurtosis(torch.cat(outputs).detach().flatten().float().cpu()))  
        print("kurtosis_output_mean:", torch.mean(torch.tensor([kurtosis(o.detach().flatten().float().cpu()) for o in outputs])))
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        for b in self.model.decoder.decode_layers:
            b.MLP2.register_forward_hook(hook_fn);
            
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        a = torch.from_numpy(preds).permute(2,1,0).reshape(44, -1)
        b = torch.from_numpy(trues).permute(2,1,0).reshape(44, -1)
        r2 = r2_score(a, b).item()
        print(r2)

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        outputs = [b.MLP2.output for b in self.model.decoder.decode_layers]  
        Inf_norm_output=max([o.abs().max() for o in outputs])
        kurtosis_output=kurtosis(torch.cat(outputs).detach().flatten().float().cpu())  
        kurtosis_output_mean=torch.mean(torch.tensor([kurtosis(o.detach().flatten().float().cpu()) for o in outputs]))
        return mae, mse, rmse, mape, mspe, Inf_norm_output, kurtosis_output, kurtosis_output_mean
