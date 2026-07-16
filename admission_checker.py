#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
admission_checker.py
=====================

Симулятор конкурсного зачисления в вуз по нескольким спискам приоритетов.

Решает задачу, которую абитуриенты обычно делают вручную через Ctrl+F по
десятку вкладок: определяет, кто из людей "выше тебя" в конкурсном списке
одной специальности реально там останется, а кто уйдёт на специальность
с более высоким приоритетом (и, соответственно, освободит место).

Алгоритм: deferred acceptance (Gale-Shapley) со стороны абитуриентов —
каждый "предлагает себя" сначала на специальность с приоритетом 1; если
не проходит по баллам — откатывается на следующий приоритет, и так пока
распределение не станет стабильным. Это ровно то, что происходит в
реальной системе вуза при перебросе мест между волнами/приоритетами.

Автор: сделано с помощью Claude (Anthropic) в роли Джонни Сильверхенда.
Лицензия: MIT (см. LICENSE)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Цветной вывод в консоль. На Windows cmd/PowerShell без colorama ANSI-коды
# иногда не работают "из коробки" - поэтому colorama опциональна, и при
# отсутствии просто откатываемся на обычный текст без цвета.
# ---------------------------------------------------------------------------
try:
    import colorama

    colorama.init()
    _COLOR_OK = True
except ImportError:
    _COLOR_OK = False


class C:
    """ANSI-цвета для терминала. Если colorama недоступна и цвет не
    поддерживается - все коды превращаются в пустые строки."""

    GREEN = "\033[92m" if _COLOR_OK or os.name != "nt" else ""
    RED = "\033[91m" if _COLOR_OK or os.name != "nt" else ""
    YELLOW = "\033[93m" if _COLOR_OK or os.name != "nt" else ""
    CYAN = "\033[96m" if _COLOR_OK or os.name != "nt" else ""
    BOLD = "\033[1m" if _COLOR_OK or os.name != "nt" else ""
    RESET = "\033[0m" if _COLOR_OK or os.name != "nt" else ""


REQUIRED_COLUMNS = ["Порядковый номер", "Приоритет конкурса", "Код поступающего"]


@dataclass
class Database:
    """Один конкурсный список (одна специальность / одна конкурсная группа)."""

    key: str            # уникальный ключ = имя файла без расширения
    spec_code: str       # код специальности (часть имени файла до последнего "_")
    seats: int           # количество бюджетных мест
    df: pd.DataFrame     # сырые данные из CSV
    source_file: str     # путь к исходному файлу


@dataclass
class SimulationResult:
    """Результат симуляции распределения по всем базам."""

    final_assignment: Dict[int, Optional[str]]
    choices: Dict[int, List[Tuple[int, str]]]
    positions: Dict[Tuple[int, str], int]
    databases: Dict[str, Database]


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_databases(data_dir: str) -> Dict[str, Database]:
    """Загружает все CSV из папки. Формат имени файла: код_колво.csv
    Пример: 09.03.01_30.csv -> специальность 09.03.01, 30 мест.

    Если у нескольких конкурсных групп совпадает код специальности (напр.
    общий конкурс и особая квота), различай их суффиксом в самом коде,
    например: 09.03.01_16.csv и "09.03.01(1)_27.csv" - ключом всё равно
    служит имя файла целиком, так что коллизии не будет в любом случае.
    """
    databases: Dict[str, Database] = {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not files:
        raise FileNotFoundError(f"В папке '{data_dir}' не найдено ни одного CSV файла.")

    for filepath in files:
        filename = os.path.splitext(os.path.basename(filepath))[0]

        if "_" not in filename:
            print(f"{C.YELLOW}[пропуск]{C.RESET} {filename}.csv — имя не в формате код_колво")
            continue

        spec_code, seats_str = filename.rsplit("_", 1)
        try:
            seats = int(seats_str)
        except ValueError:
            print(f"{C.YELLOW}[пропуск]{C.RESET} {filename}.csv — после '_' должно быть число мест")
            continue

        try:
            df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001 - хотим показать любую ошибку чтения
            print(f"{C.RED}[ошибка]{C.RESET} не удалось прочитать {filename}.csv: {exc}")
            continue

        df.columns = [c.strip().strip('"') for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"{C.YELLOW}[пропуск]{C.RESET} {filename}.csv — не хватает колонок: {missing}")
            continue

        df["Код поступающего"] = df["Код поступающего"].astype(int)
        df["Приоритет конкурса"] = df["Приоритет конкурса"].astype(int)
        df["Порядковый номер"] = df["Порядковый номер"].astype(int)

        databases[filename] = Database(
            key=filename,
            spec_code=spec_code,
            seats=seats,
            df=df,
            source_file=filepath,
        )

    if not databases:
        raise ValueError("Не удалось загрузить ни одной корректной базы.")

    return databases


def build_applicant_choices(
    databases: Dict[str, Database]
) -> Tuple[Dict[int, List[Tuple[int, str]]], Dict[Tuple[int, str], int]]:
    """Для каждого абитуриента собирает список (приоритет, ключ_базы),
    отсортированный по приоритету (меньше = желаннее). Если приоритета 1
    для человека нигде нет - сортировка сама начнёт список с минимального
    доступного приоритета, без дополнительной логики."""
    choices: Dict[int, List[Tuple[int, str]]] = {}
    positions: Dict[Tuple[int, str], int] = {}

    for db_key, database in databases.items():
        cols = list(database.df.columns)
        idx_code = cols.index("Код поступающего")
        idx_priority = cols.index("Приоритет конкурса")
        idx_position = cols.index("Порядковый номер")
        for row in database.df.itertuples(index=False, name=None):
            code = int(row[idx_code])
            priority = int(row[idx_priority])
            position = int(row[idx_position])
            choices.setdefault(code, []).append((priority, db_key))
            positions[(code, db_key)] = position

    for code in choices:
        choices[code].sort(key=lambda item: item[0])

    return choices, positions


def run_deferred_acceptance(
    databases: Dict[str, Database],
    choices: Dict[int, List[Tuple[int, str]]],
    positions: Dict[Tuple[int, str], int],
) -> Dict[int, Optional[str]]:
    """Симуляция распределения мест алгоритмом отложенного принятия
    (Gale-Shapley со стороны абитуриентов)."""
    next_choice_idx = {code: 0 for code in choices}
    current_spec: Dict[int, Optional[str]] = {code: None for code in choices}

    changed = True
    while changed:
        changed = False

        for code in choices:
            if current_spec[code] is None and next_choice_idx[code] < len(choices[code]):
                _, db_key = choices[code][next_choice_idx[code]]
                current_spec[code] = db_key

        proposals: Dict[str, List[int]] = {}
        for code, db_key in current_spec.items():
            if db_key is not None:
                proposals.setdefault(db_key, []).append(code)

        new_current_spec = dict(current_spec)
        for db_key, applicants in proposals.items():
            seats = databases[db_key].seats
            applicants_sorted = sorted(applicants, key=lambda c: positions[(c, db_key)])
            keep = set(applicants_sorted[:seats])
            for code in applicants_sorted:
                if code not in keep:
                    next_choice_idx[code] += 1
                    new_current_spec[code] = None
                    changed = True

        current_spec = new_current_spec

    return current_spec


def simulate(data_dir: str) -> SimulationResult:
    """Полный прогон: загрузка баз + симуляция распределения."""
    databases = load_databases(data_dir)
    choices, positions = build_applicant_choices(databases)
    final_assignment = run_deferred_acceptance(databases, choices, positions)
    return SimulationResult(
        final_assignment=final_assignment,
        choices=choices,
        positions=positions,
        databases=databases,
    )


# ---------------------------------------------------------------------------
# Отчёты
# ---------------------------------------------------------------------------

def print_loaded_summary(databases: Dict[str, Database]) -> None:
    print(f"\n{C.BOLD}Загруженные базы:{C.RESET}")
    for db_key, database in databases.items():
        print(f"  {C.CYAN}{db_key}{C.RESET}: {database.seats} мест, {len(database.df)} заявлений")


def print_my_priorities(my_code: int, result: SimulationResult) -> None:
    if my_code not in result.choices:
        print(f"\n{C.RED}Код {my_code} не найден ни в одной из загруженных баз.{C.RESET}")
        return

    print(f"\n{C.BOLD}Твои приоритеты (от самого желанного):{C.RESET}")
    for priority, db_key in result.choices[my_code]:
        pos = result.positions[(my_code, db_key)]
        seats = result.databases[db_key].seats
        raw_status = "хватает баллов" if pos <= seats else "не хватает баллов (без учёта переброски)"
        print(f"  приоритет {priority}: {db_key} — позиция {pos} из {seats} мест ({raw_status})")


def print_final_verdict(my_code: int, result: SimulationResult) -> None:
    my_result = result.final_assignment.get(my_code)
    print("\n" + "=" * 60)
    if my_result:
        print(f"{C.GREEN}{C.BOLD}РЕЗУЛЬТАТ: проходишь на '{my_result}'{C.RESET}")
    else:
        print(f"{C.RED}{C.BOLD}РЕЗУЛЬТАТ: пока не проходишь ни на одну специальность{C.RESET}")
        print("(из тех, чьи базы были загружены)")
    print("=" * 60)


def print_spec_detail(target: str, my_code: int, result: SimulationResult) -> None:
    if target not in result.databases:
        print(f"\n{C.RED}Специальность '{target}' не найдена среди загруженных баз.{C.RESET}")
        return

    seats = result.databases[target].seats
    assigned_here = [c for c, s in result.final_assignment.items() if s == target]
    assigned_here.sort(key=lambda c: result.positions[(c, target)])

    print(f"\n{C.BOLD}Детали по специальности '{target}' ({seats} мест):{C.RESET}")
    for i, code in enumerate(assigned_here, 1):
        marker = f"  {C.YELLOW}<-- ТЫ{C.RESET}" if code == my_code else ""
        if i <= seats:
            status = f"{C.GREEN}проходит{C.RESET}"
        else:
            status = f"{C.RED}не проходит{C.RESET}"
        print(f"  {i}. код {code} — {status}{marker}")


def print_mobility_report(target: str, my_code: int, result: SimulationResult) -> None:
    """Отчёт 'кто ушёл и куда': люди, которые изначально стояли выше тебя
    в исходном (сыром) списке target, но по итогам симуляции оказались
    приписаны к другой специальности."""
    if target not in result.databases:
        print(f"\n{C.RED}Специальность '{target}' не найдена среди загруженных баз.{C.RESET}")
        return

    if (my_code, target) not in result.positions:
        print(f"\n{C.RED}Твой код не найден в базе '{target}'.{C.RESET}")
        return

    my_raw_position = result.positions[(my_code, target)]
    df = result.databases[target].df
    above_me = df[df["Порядковый номер"] < my_raw_position]

    moved: List[Tuple[int, int, Optional[str]]] = []
    stayed: List[Tuple[int, int]] = []

    cols = list(above_me.columns)
    idx_code = cols.index("Код поступающего")
    idx_position = cols.index("Порядковый номер")
    for row in above_me.itertuples(index=False, name=None):
        code = int(row[idx_code])
        raw_pos = int(row[idx_position])
        final_spec = result.final_assignment.get(code)
        if final_spec == target:
            stayed.append((code, raw_pos))
        else:
            moved.append((code, raw_pos, final_spec))

    print(f"\n{C.BOLD}Отчёт о переброске выше тебя в '{target}' "
          f"(исходно выше тебя: {len(above_me)} чел.):{C.RESET}")
    print(f"  {C.GREEN}Ушли на другую специальность: {len(moved)}{C.RESET}")
    for code, raw_pos, final_spec in moved:
        dest = final_spec if final_spec else f"{C.RED}никуда не прошли{C.RESET}"
        print(f"    код {code} (был на {raw_pos}-м месте здесь) -> {dest}")

    print(f"  {C.YELLOW}Остались здесь: {len(stayed)}{C.RESET}")
    for code, raw_pos in stayed:
        print(f"    код {code} (был на {raw_pos}-м месте здесь) -> остаётся тут")


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------

def export_json(result: SimulationResult, path: str) -> None:
    payload = {
        "final_assignment": {
            str(code): spec for code, spec in result.final_assignment.items()
        },
        "databases": {
            key: {"seats": db.seats, "spec_code": db.spec_code, "applicants": len(db.df)}
            for key, db in result.databases.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n{C.CYAN}Экспортировано в JSON: {path}{C.RESET}")


def export_csv(result: SimulationResult, path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Код поступающего", "Итоговая специальность"])
        for code, spec in sorted(result.final_assignment.items()):
            writer.writerow([code, spec or "не поступил"])
    print(f"\n{C.CYAN}Экспортировано в CSV: {path}{C.RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admission_checker",
        description="Симулятор конкурсного зачисления в вуз по нескольким базам приоритетов.",
    )
    parser.add_argument("data_dir", help="Папка с CSV-базами (формат имени: код_колво.csv)")
    parser.add_argument("--my-code", type=int, required=True,
                         help="Твой код поступающего (обязательный параметр, нигде не хранится)")
    parser.add_argument("--target", default=None,
                         help="Ключ (имя файла без .csv) специальности для детального отчёта")
    parser.add_argument("--mobility", action="store_true",
                         help="Показать отчёт 'кто ушёл выше тебя и куда' для --target")
    parser.add_argument("--all", action="store_true",
                         help="Показать детальный отчёт по ВСЕМ загруженным специальностям")
    parser.add_argument("--export-json", metavar="PATH", default=None,
                         help="Сохранить итоговое распределение в JSON")
    parser.add_argument("--export-csv", metavar="PATH", default=None,
                         help="Сохранить итоговое распределение в CSV")
    return parser


DEFAULT_MY_CODE = None  # намеренно пусто: свой код передавай через --my-code,
                         # не храни его в коде, который может улететь на GitHub


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    my_code = args.my_code

    try:
        result = simulate(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{C.RED}Ошибка: {exc}{C.RESET}")
        return 1

    print_loaded_summary(result.databases)
    print_my_priorities(my_code, result)

    if my_code not in result.choices:
        return 1

    print_final_verdict(my_code, result)

    if args.all:
        for db_key in result.databases:
            print_spec_detail(db_key, my_code, result)
    elif args.target:
        print_spec_detail(args.target, my_code, result)
        if args.mobility:
            print_mobility_report(args.target, my_code, result)

    if args.export_json:
        export_json(result, args.export_json)
    if args.export_csv:
        export_csv(result, args.export_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
