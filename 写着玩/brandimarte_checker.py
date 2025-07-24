# 生成 Brandimarte FJSP 数据结构检查器与可视化解包器


# ================================================
# 文件名：brandimarte_checker.py
# 功能：解析并可视化 Brandimarte FJSP 数据结构
# ================================================
import sys
import os

def load_brandimarte(filepath):
    jobs = []
    with open(filepath) as f:
        header = list(map(int, f.readline().strip().split()))
        if len(header) >= 2:
            J, M = header[:2]
        else:
            raise ValueError("首行格式错误，应包含作业数和机器数")
        for j in range(J):
            line = ''
            while line.strip() == '':
                line = f.readline()
            parts = list(map(int, line.strip().split()))
            op_cnt = parts[0]
            idx = 1
            ops = []
            for _ in range(op_cnt):
                if idx >= len(parts):
                    raise ValueError(f"[错误] 第{j}个作业数据不足，当前idx={idx}")
                nmach = parts[idx]; idx += 1
                machines = []
                for _ in range(nmach):
                    if idx+1 >= len(parts):
                        raise ValueError(f"[错误] 第{j}个作业第{len(ops)}步数据不全")
                    m = parts[idx]; t = parts[idx+1]
                    machines.append((m, t))
                    idx += 2
                ops.append(machines)
            jobs.append(ops)
    return jobs, M

def check_structure(jobs):
    print(f"总作业数: {len(jobs)}")
    for j, job in enumerate(jobs):
        print(f"  作业 {j}: 共 {len(job)} 步")
        for s, step in enumerate(job):
            print(f"    步骤 {s}: 可选机器 {[(m,t) for m,t in step]}")

def main():

    jobs, M = load_brandimarte("Brandimarte_Mk01.fjs")
    print(f"机器总数: {M}")
    check_structure(jobs)

if __name__ == "__main__":
    main()
