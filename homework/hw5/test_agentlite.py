# test_agentlite.py

import os
import tempfile

import agentlite


def test_write_file():

    result = agentlite.write_file(
        "demo.txt",
        "Hello World"
    )

    assert result == "寫入成功"

    path = os.path.join(
        agentlite.WORKSPACE,
        "demo.txt"
    )

    assert os.path.exists(path)

    print("write_file PASS")


def test_read_file():

    agentlite.write_file(
        "read.txt",
        "test content"
    )

    result = agentlite.read_file(
        "read.txt"
    )

    assert result == "test content"

    print("read_file PASS")


def test_whitelist_command():

    result = agentlite.run_command(
        "echo hello"
    )

    assert "hello" in result.lower()

    print("whitelist_command PASS")


def test_block_command():

    result = agentlite.run_command(
        "rm -rf /"
    )

    assert result == "指令不允許"

    print("block_command PASS")


def test_tool_router():

    tool = {
        "tool":"write_file",
        "file":"router.txt",
        "content":"router test"
    }

    result = agentlite.tool_router(tool)

    assert result == "寫入成功"

    print("tool_router PASS")


def test_memory():

    agentlite.MEMORY.clear()

    agentlite.MEMORY.append(
        "你好"
    )

    assert len(agentlite.MEMORY) == 1

    print("memory PASS")


def main():

    tests = [
        test_write_file,
        test_read_file,
        test_whitelist_command,
        test_block_command,
        test_tool_router,
        test_memory
    ]

    passed = 0

    for test in tests:

        try:
            test()
            passed += 1

        except Exception as e:
            print(
                f"{test.__name__} FAIL"
            )
            print(e)

    print(
        f"\n通過 {passed}/{len(tests)}"
    )


if __name__ == "__main__":
    main()