import itertools

# 定义工件、工序及可选机器与加工时间
jobs_data = {
    'J1': [('O11', {'M1': 3, 'M2': 2}),
           ('O12', {'M2': 4, 'M3': 2}),
           ('O13', {'M1': 3, 'M3': 4})],
    'J2': [('O21', {'M1': 2, 'M3': 3}),
           ('O22', {'M2': 3, 'M3': 1}),
           ('O23', {'M1': 2, 'M2': 3})],
    'J3': [('O31', {'M2': 4, 'M3': 3}),
           ('O32', {'M1': 3, 'M2': 2}),
           ('O33', {'M1': 2, 'M3': 2})]
}

all_machines = ['M1', 'M2', 'M3']

# 构造所有机器选择组合（每个工序2种选择）
all_operations = []
machine_choices = []
for job, steps in jobs_data.items():
    for op_name, machine_time in steps:
        all_operations.append((job, op_name))
        machine_choices.append(list(machine_time.items()))

all_assignments = list(itertools.product(*machine_choices))

# 调度器，按作业顺序，机器先到先服务，严格模拟非并行调度
def simulate_schedule(machine_assignment):
    op_to_machine_time = {}
    for (job, op), (machine, time) in zip(all_operations, machine_assignment):
        op_to_machine_time[op] = (machine, time)

    machine_available = {m: 0 for m in all_machines}
    job_last_finish = {j: 0 for j in jobs_data.keys()}
    machine_schedules = {m: [] for m in all_machines}
    op_times = {}
    switch_cost = {m: 0 for m in all_machines}
    last_job_on_machine = {m: None for m in all_machines}

    for job, steps in jobs_data.items():
        for op_name, _ in steps:
            machine, duration = op_to_machine_time[op_name]
            ready_time = max(machine_available[machine], job_last_finish[job])
            start_time = ready_time
            end_time = start_time + duration

            # 更新机器与作业状态
            machine_available[machine] = end_time
            job_last_finish[job] = end_time
            machine_schedules[machine].append((start_time, end_time, job, op_name))
            op_times[op_name] = end_time

            # 切换成本
            if last_job_on_machine[machine] is not None and last_job_on_machine[machine] != job:
                switch_cost[machine] += 1
            last_job_on_machine[machine] = job

    makespan = max(op_times.values())
    machine_load = max(
        sum(end - start for start, end, _, _ in schedule)
        for schedule in machine_schedules.values()
    )
    total_switch = sum(switch_cost.values())

    return makespan, machine_load, total_switch

# 计算所有组合
results_fixed = []
for assign in all_assignments:
    mk, ml, sw = simulate_schedule(assign)
    results_fixed.append((mk, ml, sw))

# 提取Pareto前沿
def is_dominated(sol, all_sols):
    return any((o[0] <= sol[0] and o[1] <= sol[1] and o[2] <= sol[2]) and o != sol for o in all_sols)

pareto_front_fixed = [r for r in results_fixed if not is_dominated(r, results_fixed)]
pareto_front_fixed = sorted(pareto_front_fixed)
print(pareto_front_fixed)

# 继续补全调度器并提取符合 Pareto 前沿的调度记录
def get_schedule(machine_assignment):
    op_to_machine_time = {}
    for (job, op), (machine, time) in zip(all_operations, machine_assignment):
        op_to_machine_time[(job, op)] = (machine, time)

    machine_available = {m: 0 for m in all_machines}
    job_last_finish = {j: 0 for j in jobs_data.keys()}
    machine_schedules = {m: [] for m in all_machines}
    op_schedule = []

    switch_cost = {m: 0 for m in all_machines}
    last_job_on_machine = {m: None for m in all_machines}

    for job, steps in jobs_data.items():
        for op_name, _ in steps:
            machine, duration = op_to_machine_time[(job, op_name)]
            ready_time = max(machine_available[machine], job_last_finish[job])
            start_time = ready_time
            end_time = start_time + duration

            machine_available[machine] = end_time
            job_last_finish[job] = end_time
            machine_schedules[machine].append((start_time, end_time, job, op_name))
            op_schedule.append((job, op_name, machine, start_time, end_time))

            if last_job_on_machine[machine] is not None and last_job_on_machine[machine] != job:
                switch_cost[machine] += 1
            last_job_on_machine[machine] = job

    makespan = max(end for _, _, _, _, end in op_schedule)
    max_load = max(
        sum(e - s for s, e, _, _ in machine_schedules[m])
        for m in all_machines
    )
    total_switch = sum(switch_cost.values())

    return (makespan, max_load, total_switch), op_schedule

# 查找所有满足 Pareto 前沿解的调度方案
pareto_schedules = []
for assign in all_assignments:
    metrics, schedule = get_schedule(assign)
    if metrics in pareto_front_fixed:
        pareto_schedules.append((metrics, schedule))


print(pareto_schedules)