# XSPI1_HSLV / XSPI2_HSLV option bytes required for boot on Nucleo-H7S3L8

**Feeds:** hardware bring-up for `nucleo-h7s3l8-ssm-mamba-asd` (ssm-mcu-asd-deploy).

Fresh NUCLEO-H7S3L8 boards ship from the factory with the XSPI1_HSLV and
XSPI2_HSLV option bytes unset. Without them, `BOOT_Application()` fails at
`MapMemory()` with `BOOT_ERROR_MAPPEDMODEFAIL` (status 3) -- clocks, the XSPI2
peripheral, and UART all init fine; only the memory-mapped XIP handshake with
the external flash fails. This is a known, common first-bring-up issue,
confirmed across multiple independent reports, not specific to this project's
configuration.

**Fix:** STM32CubeProgrammer > connect via ST-LINK > Option Bytes panel >
enable both XSPI1_HSLV and XSPI2_HSLV > Apply > power-cycle the board. One-time
per physical board; persists across reflashing.