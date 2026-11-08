"""Unicorn oracle: run the native treelite code from libalgos.so."""
import math, os, struct
import numpy as np
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm64_const import *

SO = os.environ["NSSA_LIBALGOS"]  # set by tools/extract_nssa_classifier.py
PAGE = 0x1000
STACK = 0x7000_0000
DATA = 0x7100_0000
RES = 0x7200_0000
STUB = 0x7300_0000
RET = 0x7300_1000


class Native:
    def __init__(self):
        f = open(SO, 'rb')
        elf = ELFFile(f)
        uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        hi = 0
        for seg in elf.iter_segments():
            if seg['p_type'] != 'PT_LOAD':
                continue
            va = seg['p_vaddr'] & ~(PAGE - 1)
            end = (seg['p_vaddr'] + seg['p_memsz'] + PAGE - 1) & ~(PAGE - 1)
            if end > hi:
                try:
                    uc.mem_map(max(va, hi), end - max(va, hi))
                except Exception:
                    pass
                hi = end
            uc.mem_write(seg['p_vaddr'], seg.data())
        self.syms = {}
        for s in elf.get_section_by_name('.dynsym').iter_symbols():
            if s['st_value']:
                self.syms[s.name] = s['st_value']
        dynsym = elf.get_section_by_name('.dynsym')
        stubs = {}
        for sec in elf.iter_sections():
            if isinstance(sec, RelocationSection) and sec.name == '.rela.plt':
                for r in sec.iter_relocations():
                    sym = dynsym.get_symbol(r['r_info_sym'])
                    if sym['st_value']:
                        tgt = sym['st_value']
                    else:
                        tgt = STUB + 4 * len(stubs)
                        stubs[tgt] = sym.name
                    uc.mem_write(r['r_offset'], struct.pack('<Q', tgt))
        self.stubs = stubs
        uc.mem_map(STACK, 0x100000)
        uc.mem_map(DATA, 0x10000)
        uc.mem_map(RES, 0x1000)
        uc.mem_map(STUB, 0x2000)
        uc.mem_write(STUB, b'\xc0\x03\x5f\xd6' * 0x800)   # ret everywhere
        uc.hook_add(UC_HOOK_CODE, self._stub, begin=STUB, end=STUB + 0x1000)
        uc.mem_write(RET, b'\x1f\x20\x03\xd5')    # nop
        self.uc = uc
        # enable FP/SIMD (CPACR) -- unicorn arm64 enables it by default

    def _stub(self, uc, addr, size, user):
        name = self.stubs.get(addr)
        if name == 'exp':
            d = uc.reg_read(UC_ARM64_REG_D0)
            x = struct.unpack('<d', struct.pack('<Q', d))[0]
            try:
                y = math.exp(x)
            except OverflowError:
                y = float('inf')
            uc.reg_write(UC_ARM64_REG_D0, struct.unpack('<Q', struct.pack('<d', y))[0])
        elif name == '__stack_chk_fail':
            raise RuntimeError('stack chk')
        elif name is None:
            raise RuntimeError('stub %x' % addr)
        else:
            raise RuntimeError('unexpected call %s' % name)

    def call(self, fn, data_bits, res_bits, x1=RES, x2=None, w1=None):
        uc = self.uc
        uc.mem_write(DATA, b''.join(struct.pack('<Q', int(b)) for b in data_bits))
        uc.mem_write(RES, b''.join(struct.pack('<Q', int(b)) for b in res_bits))
        uc.reg_write(UC_ARM64_REG_SP, STACK + 0x80000)
        uc.reg_write(UC_ARM64_REG_X0, DATA)
        uc.reg_write(UC_ARM64_REG_X30, RET)
        uc.reg_write(UC_ARM64_REG_TPIDR_EL0, STACK + 0xF0000)
        if w1 is None:
            uc.reg_write(UC_ARM64_REG_X1, x1)
        else:
            uc.reg_write(UC_ARM64_REG_X1, w1)
            uc.reg_write(UC_ARM64_REG_X2, x2)
        uc.emu_start(self.syms[fn], RET)
        out = uc.mem_read(RES, 32)
        return list(struct.unpack('<4Q', bytes(out)))

    def unit(self, i, data_bits, res_bits):
        return self.call('predict_margin_multiclass_unit%d' % i, data_bits, res_bits)

    def predict(self, data_bits, pred_margin=0):
        return self.call('predict_multiclass', data_bits, [0, 0, 0, 0], x2=RES, w1=pred_margin)


def entries_from_row(row):
    """Mirror nssaops::sleep_classifier: NaN -> missing (-1 in the low word, high 0)."""
    out = []
    for v in row:
        f = np.float32(v)
        if f != f:
            out.append(0x00000000FFFFFFFF)
        else:
            out.append(struct.unpack('<Q', struct.pack('<d', float(f)))[0])
    return out
