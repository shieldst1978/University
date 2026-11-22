"""CPU Scheduling and Memory Allocation Simulator

CSO7002 Mid-Module Assessment
Student name: Tim Shields
Student number: 2517896

From this script's directory, run ``python 2517896_scheduler.py``. The program uses
built-in process data and prints the simulation results to the terminal.
Priority scheduling decisions are shown by default or with ``--trace``.
Run ``python 2517896_scheduler.py --no-trace`` to hide those decisions.

Assumptions:
- Ties beyond arrival time are broken by input order.
- Elapsed time is measured from t=0.
- Memory is allocated once, in input order, and never released.
- Partitions are numbered from 1 in the output.
"""

from dataclasses import dataclass
import argparse
import copy


@dataclass
class Process:
    """Represent a process with a simplified process control block (PCB)."""

    pid: str
    arrival: int
    burst: int
    memory_kb: int
    priority: int = 0   # lower number = higher priority
    start: int = 0
    finish: int = 0


PARTITIONS_KB = [300, 150, 100, 250, 200, 120]


def schedule_sjf(processes):
    """Non-preemptive Shortest Job First (SJF).
    Break ties by arrival time, then input order. The stable arrival sort
    preserves input order for simultaneous arrivals, and min returns the
    first process when the selection keys are equal.
    """
    time = 0          # current simulation time
    done = []         # processes in execution order
    waiting = sorted(processes, key=lambda p: p.arrival)

    while waiting:
        ready = [p for p in waiting if p.arrival <= time]
        if not ready:     # CPU idle until next arrival
            time = waiting[0].arrival
            continue

        job = min(ready,
                  key=lambda p: (p.burst, p.arrival))
        waiting.remove(job)
        job.start = time
        # Non-preemptive: the selected job runs for its entire CPU burst.
        job.finish = job.start + job.burst
        time = job.finish
        done.append(job)

    return done


def schedule_priority(processes, *, trace=False):
    """Run each selected process to completion. Reduce its priority number
    by one per five complete units waiting, to a minimum of 1.
    Break ties by arrival time, then input order. The stable arrival sort
    and ready-list insertion order preserve input order for simultaneous
    arrivals, and min returns the first process with the lowest key.
    """
    time = 0
    done = []
    waiting = sorted(processes, key=lambda p: p.arrival)
    ready = []
    next_job = 0

    while ready or next_job < len(waiting):
        if not ready:
            time = max(time, waiting[next_job].arrival)

        while next_job < len(waiting) and waiting[next_job].arrival <= time:
            ready.append(waiting[next_job])
            next_job += 1

        # Effective priority: original minus complete 5-unit waits since arrival (min 1)
        ranked = [
            (max(1, p.priority - (time - p.arrival) // 5), p)
            for p in ready
        ]
        _, job = min(ranked, key=lambda entry: (entry[0], entry[1].arrival))
        if trace:
            ready_priorities = ", ".join(
                f"{process.pid}={priority}" for priority, process in ranked
            )
            print(f"t={time} | Ready (effective priority) {ready_priorities}"
                  f" | Selected {job.pid}")
        ready.remove(job)
        job.start = time
        job.finish = time + job.burst
        time = job.finish
        done.append(job)

    return done


def calculate_metrics(done):
    """Return timing and performance metrics for a completed schedule."""
    waiting = sum(job.start - job.arrival for job in done) / len(done) if done else 0
    turnaround = sum(job.finish - job.arrival for job in done) / len(done) if done else 0
    elapsed = done[-1].finish if done else 0
    busy = sum(job.burst for job in done)
    utilisation = busy / elapsed if elapsed else 0
    throughput = len(done) / elapsed if elapsed else 0
    return {
        "average_waiting_time": waiting,
        "average_turnaround_time": turnaround,
        "elapsed_time": elapsed,
        "idle_time": elapsed - busy,
        "cpu_utilisation": utilisation,
        "throughput": throughput,
    }


def allocate_memory(processes, strategy):
    """Place each process in a fixed partition (one
    process per partition). strategy is "first" or
    "best". Return {pid: (partition index,
    internal fragmentation in KB)}, or None for a
    process that cannot be allocated."""
    free = list(enumerate(PARTITIONS_KB))
    if strategy == "best":
        free.sort(key=lambda partition: partition[1])
    elif strategy != "first":
        raise ValueError(f"Unknown memory allocation strategy: {strategy!r}")

    allocations = {}
    for process in processes:
        required = process.memory_kb
        for position, (index, size) in enumerate(free):
            if size >= required:
                allocations[process.pid] = (index, size - required)
                free.pop(position)
                break
        else:
            allocations[process.pid] = None

    return allocations


def display_results(name, done, metrics):
    """Display the schedule and metrics."""
    print("\n" + name)
    row_format = "{:<6}  {:>7}  {:>5}  {:>6}  {:>7}  {:>10}"
    print(row_format.format("PID", "Arrival", "Start", "Finish",
                            "Waiting", "Turnaround"))
    for p in done:
        print(row_format.format(p.pid, p.arrival, p.start, p.finish,
                                p.start - p.arrival, p.finish - p.arrival))
    print(f"\nTotal elapsed time: {metrics['elapsed_time']} time units")
    print(f"CPU idle time: {metrics['idle_time']} time units")
    print(f"Average waiting time: {metrics['average_waiting_time']:.2f}")
    print(f"Average turnaround time: {metrics['average_turnaround_time']:.2f}")
    print(f"CPU utilisation: {metrics['cpu_utilisation']:.2%}")
    print(f"Throughput: {metrics['throughput']:.3f} processes per time unit")


def run_scheduling(processes, *, trace=False):
    """Run SJF and priority scheduling with ageing, then print their metrics."""
    done = schedule_sjf(copy.deepcopy(processes))
    display_results("SJF (non-preemptive)", done,
                    calculate_metrics(done))

    if trace:
        print("\nPriority scheduling trace")
    done = schedule_priority(copy.deepcopy(processes), trace=trace)
    display_results("Priority (non-preemptive, with ageing)", done,
                    calculate_metrics(done))


def run_memory_allocation(processes):
    """Print first-fit and best-fit allocations and their internal fragmentation."""
    for strategy in ("first", "best"):
        allocations = allocate_memory(processes, strategy)
        allocated_count = 0
        total_fragmentation = 0
        print(f"\nMemory allocation ({strategy} fit)")
        row_format = "{:<6}  {:<11}  {:>9}  {:>27}"
        print(row_format.format("PID", "Partition", "Size (KB)",
                                "Internal fragmentation (KB)"))
        for pid, allocation in allocations.items():
            if allocation is None:
                print(row_format.format(pid, "Unallocated", "-", "-"))
            else:
                partition_index, fragmentation = allocation
                allocated_count += 1
                total_fragmentation += fragmentation
                print(row_format.format(pid, partition_index + 1,
                                        PARTITIONS_KB[partition_index], fragmentation))
        print(f"Allocated {allocated_count} of {len(allocations)} processes | "
              f"Total internal fragmentation: {total_fragmentation} KB")


def main(*, trace=True):
    """Run the built-in scheduling and memory allocation scenarios."""
    processes = [
        Process("P1", 0, 9, 180, 3),
        Process("P2", 1, 3, 90, 1),
        Process("P3", 2, 6, 230, 4),
        Process("P4", 3, 3, 120, 2),
        Process("P5", 4, 8, 60, 5),
        Process("P6", 21, 4, 140, 2),
        Process("P7", 35, 5, 200, 1),
    ]
    print("\nSupplied dataset (P1-P7)")
    run_scheduling(processes, trace=trace)
    run_memory_allocation(processes)

    p8 = Process("P8", 35, 5, 301, 1)
    print("\nExtended scheduling test (P1-P8)")
    run_scheduling([*processes, p8], trace=trace)

    print("\nOversized memory request test (P8 first, 301 KB)")
    run_memory_allocation([p8, processes[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU scheduling and memory allocation simulator")
    trace_options = parser.add_mutually_exclusive_group()
    trace_options.add_argument("--trace", action="store_true",
                               help="show priority scheduling decisions (default)")
    trace_options.add_argument("--no-trace", dest="trace", action="store_false",
                               help="hide priority scheduling decisions")
    parser.set_defaults(trace=True)
    main(trace=parser.parse_args().trace)

