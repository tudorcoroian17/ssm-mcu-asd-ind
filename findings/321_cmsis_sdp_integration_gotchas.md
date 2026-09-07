# CMSIS-DSP v1.16.2 integration gotchas on STM32CubeIDE managed build

**Feeds:** CMSIS-DSP setup for `nucleo-h7s3l8-ssm-mamba-asd` Appli project (§3.2 feature pipeline).

Two build failures, both specific to how CMSIS-DSP v1.16.2 packages its source
relative to a plain `Include` + `Source` copy:

1. **Missing `PrivateInclude`.** CMSIS-DSP splits headers into `Include`
   (public API) and `PrivateInclude` (three headers -- `arm_sorting.h`,
   `arm_vec_fft.h`, `arm_vec_filtering.h` -- needed only to compile the
   library's own `.c` files, not part of the public API). Copying just
   `Include` and `Source` produces `fatal error: arm_sorting.h: No such file
   or directory` in `SupportFunctions`. Fix: copy `PrivateInclude` alongside
   the other two, and add it as a third include path.

2. **Duplicate symbol definitions per category.** Each `Source/<Category>/`
   folder contains one aggregator file matching its own folder name (e.g.
   `Source/WindowFunctions/WindowFunctions.c`), which `#include`s every
   individual function file in that folder. Compiling both the aggregator
   and the individual files -- the default once the whole folder is a
   registered source location -- causes "multiple definition" linker errors
   for every function in that category. Fix: rename every file whose name
   matches its parent folder so the build skips it:

```powershell
   Get-ChildItem -Path ".\Appli\Middlewares\CMSIS-DSP\Source" -Recurse -Filter "*.c" |
     Where-Object { $_.BaseName -eq $_.Directory.Name } |
     Rename-Item -NewName { $_.Name + "_old" }
```

**Final working setup:** `Appli/Middlewares/CMSIS-DSP/{Include,PrivateInclude,Source}`,
registered as an Appli-only source location (not the shared `Drivers` folder,
which Boot also builds from and has no need for CMSIS-DSP). Include paths:
`../Middlewares/CMSIS-DSP/Include`, `../Middlewares/CMSIS-DSP/PrivateInclude`.
Preprocessor define: `ARM_MATH_CM7`. Verified working via `arm_sqrt_f32`.