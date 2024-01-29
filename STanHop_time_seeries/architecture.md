```
STanHopNet(
  (enc_emb): PatchEmbedding(
    (linear): Linear(in_features=6, out_features=512, bias=True)
  )
  (pre_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (encoder): Encoder(
    (encode_blocks): ModuleList(
      (0): scale_block(
        (encode_layers): ModuleList(
          (0): STHMLayer(
            (cross_time): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (cross_series): HopfieldPooling(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (hopfield): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (MLP1): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
            (MLP2): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
          )
        )
      )
      (1): scale_block(
        (merge_layer): SegMerging(
          (linear_trans): Linear(in_features=2048, out_features=512, bias=True)
          (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
        (encode_layers): ModuleList(
          (0): STHMLayer(
            (cross_time): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (cross_series): HopfieldPooling(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (hopfield): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (MLP1): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
            (MLP2): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
          )
        )
      )
      (2): scale_block(
        (merge_layer): SegMerging(
          (linear_trans): Linear(in_features=2048, out_features=512, bias=True)
          (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
        (encode_layers): ModuleList(
          (0): STHMLayer(
            (cross_time): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (cross_series): HopfieldPooling(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (hopfield): Hopfield(
              (inner_attention): Association(
                (dropout): Dropout(p=0.0, inplace=False)
                (softmax): EntmaxAlpha()
              )
              (query_projection): Linear(in_features=512, out_features=512, bias=True)
              (key_projection): Linear(in_features=512, out_features=512, bias=True)
              (value_projection): Linear(in_features=512, out_features=512, bias=True)
              (out_projection): Linear(in_features=512, out_features=512, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (MLP1): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
            (MLP2): Sequential(
              (0): Linear(in_features=512, out_features=1024, bias=True)
              (1): GELU()
              (2): Linear(in_features=1024, out_features=512, bias=True)
            )
          )
        )
      )
    )
  )
  (decoder): Decoder(
    (decode_layers): ModuleList(
      (0): DecoderLayer(
        (sthm): STHMLayer(
          (cross_time): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (cross_series): HopfieldPooling(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (hopfield): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.0, inplace=False)
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (MLP1): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
          (MLP2): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
        )
        (hopfield): Hopfield(
          (inner_attention): Association(
            (dropout): Dropout(p=0.0, inplace=False)
            (softmax): EntmaxAlpha()
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (MLP1): Linear(in_features=512, out_features=512, bias=True)
        (gelu): GELU()
        (MLP2): Linear(in_features=512, out_features=512, bias=True)
        (linear_pred): Linear(in_features=512, out_features=6, bias=True)
      )
      (1): DecoderLayer(
        (sthm): STHMLayer(
          (cross_time): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (cross_series): HopfieldPooling(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (hopfield): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.0, inplace=False)
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (MLP1): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
          (MLP2): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
        )
        (hopfield): Hopfield(
          (inner_attention): Association(
            (dropout): Dropout(p=0.0, inplace=False)
            (softmax): EntmaxAlpha()
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (MLP1): Linear(in_features=512, out_features=512, bias=True)
        (gelu): GELU()
        (MLP2): Linear(in_features=512, out_features=512, bias=True)
        (linear_pred): Linear(in_features=512, out_features=6, bias=True)
      )
      (2): DecoderLayer(
        (sthm): STHMLayer(
          (cross_time): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (cross_series): HopfieldPooling(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (hopfield): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.0, inplace=False)
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (MLP1): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
          (MLP2): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
        )
        (hopfield): Hopfield(
          (inner_attention): Association(
            (dropout): Dropout(p=0.0, inplace=False)
            (softmax): EntmaxAlpha()
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (MLP1): Linear(in_features=512, out_features=512, bias=True)
        (gelu): GELU()
        (MLP2): Linear(in_features=512, out_features=512, bias=True)
        (linear_pred): Linear(in_features=512, out_features=6, bias=True)
      )
      (3): DecoderLayer(
        (sthm): STHMLayer(
          (cross_time): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (cross_series): HopfieldPooling(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (hopfield): Hopfield(
            (inner_attention): Association(
              (dropout): Dropout(p=0.0, inplace=False)
              (softmax): EntmaxAlpha()
            )
            (query_projection): Linear(in_features=512, out_features=512, bias=True)
            (key_projection): Linear(in_features=512, out_features=512, bias=True)
            (value_projection): Linear(in_features=512, out_features=512, bias=True)
            (out_projection): Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.0, inplace=False)
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (MLP1): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
          (MLP2): Sequential(
            (0): Linear(in_features=512, out_features=1024, bias=True)
            (1): GELU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
          )
        )
        (hopfield): Hopfield(
          (inner_attention): Association(
            (dropout): Dropout(p=0.0, inplace=False)
            (softmax): EntmaxAlpha()
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (MLP1): Linear(in_features=512, out_features=512, bias=True)
        (gelu): GELU()
        (MLP2): Linear(in_features=512, out_features=512, bias=True)
        (linear_pred): Linear(in_features=512, out_features=6, bias=True)
      )
    )
  )
)
```