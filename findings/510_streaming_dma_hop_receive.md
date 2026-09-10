# 450 -- Streaming DMA hop receive on the Nucleo-H7S3L8

## Summary

The Nucleo-H7S3L8 firmware now receives audio one hop at a time over UART
using background DMA, instead of buffering a whole clip in RAM. Output is
bit-parity-equivalent to the reference embedding within the tolerance
established in finding 430.

## Problem

The Phase 3/4 transport buffered a full clip (`clip_buffer`, 360,000 B) and
its log-mel output (`logmel_output`, 90,112 B) in main RAM, leaving a 116 B
margin (finding 440). These buffers were parity-harness scaffolding, not a
deployment requirement.

## Change

- Removed both whole-clip buffers. The host pads each clip's final hop to a
  full hop with zeros and streams hops back to back; the MCU processes one
  hop per iteration and discards it.
- Reception uses `HAL_UART_Receive_DMA` on a GPDMA1 channel, double-buffered
  so the next hop's DMA overlaps the current hop's compute.
- Disabled the `SSM_SEND_LOGMEL_DEBUG` path (log-mel parity was already
  established in findings 322--325).

## Result

- Main RAM use dropped from 465,804 B (116 B free) to ~17,000 B.
- Backbone compute: ~4,678 us/frame against a 32,000 us budget (~6.8x
  headroom).
- Parity: max abs error 5.99e-4, mean abs error 1.45e-4 -- identical to the
  pre-streaming figures in finding 440.

## Gotchas

- USART3 (the `COM1` VCP UART) is BSP-controlled, so CubeMX cannot configure
  its GPDMA request through the GUI. The request must be set in application
  code (`GPDMA1_Init 2` user block) and the DMA linked with `__HAL_LINKDMA`
  after `BSP_COM_Init`.
- The GPDMA request must be `GPDMA1_REQUEST_USART3_RX`. Selecting `_TX` arms
  the channel with no error but no completion (`RxState` stays BUSY_RX,
  `ErrorCode` 0, wait times out) -- the request line never triggers.
- Cortex-M7 D-Cache must be invalidated over the DMA-written buffer before
  the CPU reads it (`SCB_InvalidateDCache_by_Addr`). This is M7-specific and
  does not apply to the RP2040 or ESP32.

<!-- Claude comment: the double-buffering here is the reusable idea for the
other two boards, but none of the mechanism ports directly -- the RP2040
needs its own DMA/PIO plumbing under the Arduino core, and the ESP32's
ESP-IDF UART driver already ring-buffers in the background, likely removing
the need for explicit DMA there. The D-Cache step is unique to this board. -->