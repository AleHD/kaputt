import re
import json
import asyncio
import warnings
from typing import Literal, Optional
from subprocess import Popen, PIPE


def is_running(jobid: str) -> bool:
    with Popen(["squeue", "-j", jobid, "--json"], stdout=PIPE, stderr=PIPE,
               text=True) as proc:
        if proc.wait() != 0:
            warnings.warn(f"Couldn't find job {jobid}. Assuming it is completed")
            return False
        json_txt = "".join(proc.stdout)
    job_info, = json.loads(json_txt)["jobs"]
    return job_info["job_state"] in {"RUNNING", "PENDING"}


def get_fail_kind(jobid: str) -> Optional[str]:
    with open(f"logs/kaputt_{jobid}.err") as f:
        contents = "".join(f)
    if len(contents) == 0:
        return None
    if "CUDA-capable device(s) is/are busy or unavailable" in contents:
        return "device busy"
    if "CUDA error: operation not supported" in contents:
        return "operation not supported"
    return "unknown"


async def has_faulty(host: str, suspects: list[str]) -> tuple[Optional[str], str]:
    nodes = len(suspects)
    nodelist = host + "[" + ",".join(suspects) + "]"
    cmd = ["sbatch",  "--nodelist", nodelist, "--nodes", str(nodes), "scripts/run.sbatch"]
    with Popen(cmd, text=True, stdout=PIPE) as proc:
        proc.wait()
        jobid, = re.match("^Submitted batch job (.*)$", proc.stdout.read().strip()).groups()
    await asyncio.sleep(2)
    while is_running(jobid):
        await asyncio.sleep(2)
    return get_fail_kind(jobid), jobid


def investigate(host: str, suspects: list[str]):
    async def dummy() -> Literal[[]]:
        return []

    async def refine(suspects: list[str]) -> list[dict[str, str]]:
        assert len(suspects) > 0
        if len(suspects) == 1:
            fail_kind, jobid = await has_faulty(host, suspects)
            is_faulty = fail_kind is not None
            if is_faulty:
                print("! Node", suspects[0], "found to be faulty!")
                return [{"node": suspects[0], "jobid": jobid, "fail": fail_kind}]
            return []

        partition1, partition2 = suspects[: len(suspects)//2], suspects[len(suspects)//2 :]
        print("? Querying for nodes", partition1)
        task1 = asyncio.create_task(has_faulty(host, partition1))
        print("? Querying for nodes", partition2)
        task2 = asyncio.create_task(has_faulty(host, partition2))

        kind1, jobid1 = await task1
        faulty1 = kind1 is not None
        if faulty1:
            print("x", jobid1, "Faulty node found in", partition1)
        else:
            print("o Healthy partition", partition1)
        kind2, jobid2 = await task2
        faulty2 = kind2 is not None
        if faulty2:
            print("x", jobid2, "Faulty node found in", partition2)
        else:
            print("o Healthy partition", partition2)

        faulty = []
        task1 = asyncio.create_task(refine(partition1) if faulty1 else dummy())
        task2 = asyncio.create_task(refine(partition2) if faulty2 else dummy())
        faulty += await task1
        faulty += await task2
        return faulty

    print("Starting investigation with", len(suspects), "suspects:", suspects)
    print("Making sure entire partition has faulty...")
    kind, jobid = asyncio.run(has_faulty(host, suspects))
    faulty = kind is not None
    if not faulty:
        warnings.warn("Entire partition didn't have faulty node. Terminating")
        return
    print("Yes, entire partition has faulty", jobid, "Refining suspects...")

    faulty_nodes = asyncio.run(refine(suspects))
    print("Investigation over!")
    print("Faulty nodes:")
    print(faulty_nodes)

    print("Just the nodes:")
    print(",".join(info["node"] for info in faulty_nodes))
