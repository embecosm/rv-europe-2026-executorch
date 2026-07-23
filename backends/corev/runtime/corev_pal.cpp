/*
 * Platform abstraction layer (PAL) override.
 *
 * Copyright (C) 2026 Embecosm Limited
 * Contributor Jeremy Bennett <jeremy.bennett@embecosm.com>
 * Contributor Shane Slattery <shane.slattery@embecosm.com>
 *
 * This file is part of the Embecosm ExecuTorch tutorial for RISC-V.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>
#include <cstdlib>
#include <stdio.h>

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
  __builtin_trap();
}

et_timestamp_t et_pal_current_ticks(void) {
  uint32_t cycle_lo, cycle_hi;
  uint64_t cycles;
  asm volatile(
      "csrr t1, mcycle\n"
      "sw t1, %0\n"
      "csrr t1, mcycleh\n"
      "sw t1, %1\n"
      : "=m"(cycle_lo), "=m"(cycle_hi)
      :
      : "t1", "memory");
  cycles = static_cast<uint64_t> (cycle_hi) << 32;
  cycles |= static_cast<uint64_t> (cycle_lo);
  return static_cast<et_timestamp_t> (cycles);
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  return {5, 2};
}

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  const char *level_name;
  switch (level)
    {
    case 'D': level_name = "Debug"; break;
    case 'I': level_name = "Info"; break;
    case 'E': level_name = "ERROR"; break;
    case 'F': level_name = "*** FATAL ERROR ***"; break;
    default: level_name = "??? UNKNOWN ???"; break;
    }
  printf ("%lu: %s %s:%s:%d %s\n", timestamp, level_name, filename, function,
	  line, message);
}

void* et_pal_allocate(size_t size) {
  return malloc(size);
}

void et_pal_free(void* ptr) {
  free(ptr);
}
