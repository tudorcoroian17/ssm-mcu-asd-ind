# UART bulk transfer: overrun flag and stale receive data

**Feeds:** MCU<->PC clip/result transport for all three boards.

**Symptom:** MCU->PC transfers of ~88 KB silently truncated at varying offsets
(18-22 KB), with `HAL_UART_Transmit` returning `HAL_OK` throughout. Pattern
tests showed a *hole* mid-stream at offset 22528 (22 x 1024), not a clean
truncation.

**Actual cause (two stacked faults, both in the RECEIVE path):**

1. **Latched ORE.** The ~352 KB polled clip upload (`HAL_UART_Receive` byte by
   byte) can set the overrun flag without failing that call -- it still returns
   the full byte count. The flag stays latched and causes the *next*
   `HAL_UART_Receive` to return `HAL_ERROR` immediately, reported as
   `err=0x00000008` (`HAL_UART_ERROR_ORE`).
2. **Stale queued bytes.** The PC pushes more bytes than the MCU reads; the
   surplus sits queued and is returned in place of the next expected message
   (seen as `ACKBAD bytes=00 FE 65 FE` -- audio samples, not the ACK).

**Fix:** after any large polled receive, before the next receive:
clear ORE/NE/FE/PE flags, reset `ErrorCode`, then drain with single-byte reads
until a short timeout elapses. On the PC side, `time.sleep(0.5)` +
`reset_output_buffer()` after sending the clip.

**Also required:** chunked transmit (4096 B) with a 4-byte `ACK!` from the PC
per chunk. Single-byte acks proved unreliable on this link.

**Claude comment:** the misleading part is that every symptom pointed at the
transmit direction, and the MCU's own status codes said everything succeeded.
Only `huart.ErrorCode` after a failure revealed the truth. Print that value
first next time, not last.