# Shared double-precision three-dimensional neural operators

This is reusable implementation code for the development campaign, with independently checked Fourier indexing and bounded training/validation smokes. It does not contain accepted accuracy or runtime claims.

`models3d.init_model(key, spec, in_channels, out_channels)` returns a parameter tree. `models3d.apply_model(params, input, spec)` takes and returns channels-last arrays of shape `(batch, nx, ny, nz, channels)`. Every parameter and activation is float64; Fourier intermediates are complex128. All three grid axes are spatial. Parameters are explicit JIT arguments; do not close over large data arrays.

The FNO configuration is `dict(kind='fno3d', width=16, modes=[6,6,6], depth=4, padding=9)`. It uses four learned signed-mode Fourier convolutions plus pointwise linear maps, GELU activations, lifting and readout. The first two mode counts specify positive and negative frequencies; the third specifies the real-FFT half-plane count. The scalar Dirichlet adapter pads on the positive edges; periodic problems must set `padding=0`.

The U-Net configuration is `dict(kind='unet3d', width=8, levels=3, periodic=False)`. It has two convolutions at each encoder/decoder level, average pooling, interpolation to the exact skip shape, concatenated skip connections and a pointwise readout. Periodic convolution, odd-size pooling and interpolation all wrap when `periodic=True`. This is a three-dimensional adaptation of the U-Net architecture, without batch normalization.

`training.train(train_x, train_y, valid_x, valid_y, spec, cfg, out, dump, checkpoint, train_denominators=None, valid_denominators=None)` returns selected parameters and training metadata. The two callback functions take `(path, value)` and persist JSON and pickle respectively. The configuration contains `steps`, `wall_seconds`, `batch_size`, `seed`, `learning_rate`, `validation_every`, `curve_every`, and `components_per_output`. Output channels pack time first and vector components second. By default each time's squared field error is divided by that output's squared vector norm. To use initial-field normalization, supply explicit positive squared norms shaped `(cases, output_times)` for both datasets and set the human-readable `normalization` field. The loss is mean relative squared error plus one tenth its squared tail penalty. Validation selects the smallest worst error over all development cases and requested output times. Final data must never enter this function.

The task adapter owns training-only physical scales, coordinates or supplied coefficients, boundary conventions, target packing, output projection and initial-field pass-through. Heat's adapter accepts the supplied interior initial field and three coordinates, returns five evolved fields, and prepends the exact supplied initial field. NS needs velocity plus supplied viscosity, periodic settings, and explicitly charged projection if used; Poisson needs forcing-to-solution packing. Dataset-generation descriptors are not automatically legitimate online inputs.

Persist the selected checkpoint, latest optimizer state, complete learning curve, actual training steps/time, cohort hashes, precision, source hashes and task-adapter provenance. Compare inference timings only within the same allocation, with burn-in and complete dense output. Training budgets and model sizes are declared per model and are not implicitly capacity matched.

## Glossary

- **FNO:** Fourier neural operator; learned low-frequency spatial convolutions combined with local channel mixing.
- **U-Net:** convolutional encoder and decoder connected by same-resolution skip features.
- **Channels last:** field component and output-time information occupies the last array axis.
- **Padding / periodic:** zero extension outside an interval / wrapping opposite domain faces.
- **Current / initial normalization:** divide by the reference field's norm at the output time / at the initial time.
- **Validation:** development data used to choose a checkpoint; separate from final evaluation data.
- **Float64 / complex128:** double-precision real / complex arithmetic.
