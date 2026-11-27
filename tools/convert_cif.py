#!/usr/bin/env python3
"""
898253.cif 处理脚本：
  1) 诊断 ASE 读取失败的原因并生成修正版 CIF（空间群 '?' -> 'P -1'）
  2) 用 pymatgen 直接将原始 CIF 转为 POSCAR
  3) 交叉验证 ASE / pymatgen 两个结构一致
运行：/home/feng/.local/anaconda/bin/python convert_cif.py
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ase.io import read as ase_read
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SRC = Path("898253.cif")
FIXED_CIF = Path("898253_fixed.cif")   # 供 ASE 读取的修正版
POSCAR = Path("898253_POSCAR.vasp")

# ---------- 1. 修正 CIF ----------
text = SRC.read_text()
old = "_symmetry_space_group_name_H-M   ?"
assert old in text, "未找到待替换的字段，请检查 CIF 格式"
# 也可写成 "_symmetry_int_tables_number  2"
text = text.replace(old, "_symmetry_space_group_name_H-M  'P -1'")
FIXED_CIF.write_text(text)

atoms = ase_read(FIXED_CIF)
print(f"[ASE]    修正版 CIF 读取成功: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

# ---------- 2. pymatgen 转 POSCAR ----------
s = CifParser(str(SRC)).parse_structures(primitive=True)[0]
spg = SpacegroupAnalyzer(s)
print(f"[pymatgen] {len(s)} atoms, {spg.get_space_group_symbol()} "
      f"(No.{spg.get_space_group_number()}), {spg.get_crystal_system()}, "
      f"V={s.lattice.volume:.3f} Å³")

Poscar(s).write_file(POSCAR)
print(f"[pymatgen] 已写出 {POSCAR}")

# ---------- 3. 验证 ----------
s_poscar = Poscar.from_file(POSCAR).structure
s_ase = AseAtomsAdaptor.get_structure(atoms)
ok = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=2).fit(s_ase, s_poscar)
print(f"[验证]    POSCAR 回读与 ASE 结构匹配: {ok}")
assert ok, "转换结果与 ASE 结构不一致"
