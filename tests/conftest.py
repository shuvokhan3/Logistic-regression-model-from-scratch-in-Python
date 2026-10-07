##conftest.py is a shared configuration/setup file for pytest.

def pytest_configure(config):
    # Module 0 markers (keep these existing lines)
    config.addinivalue_line("markers", "task0_1: Task 0.1 operators")
    config.addinivalue_line("markers", "task0_2: Task 0.2 property tests")
    config.addinivalue_line("markers", "task0_3: Task 0.3 functional")
    config.addinivalue_line("markers", "task0_4: Task 0.4 modules")
    # Module 1 markers
    config.addinivalue_line("markers", "task1_1: Task 1.1 numerical derivatives")
    config.addinivalue_line("markers", "task1_2: Task 1.2 scalar forward")
    config.addinivalue_line("markers", "task1_3: Task 1.3 chain rule")
    config.addinivalue_line("markers", "task1_4: Task 1.4 backpropagation")
    config.addinivalue_line("markers", "task1_5: Task 1.5 training")
    # Module 2 markers
    config.addinivalue_line("markers", "task2-1: Task 2.1 tensor indexing")
    config.addinivalue_line("markers", "task2_2: Task 2.2 broadcasting")
    config.addinivalue_line("markers", "task2_3: Task 2.3 tensor operations")
    config.addinivalue_line("markers", "task2_4: Task 2.4 tensor autodiff")
    config.addinivalue_line("markers", "task2_5: Task 2.5 tensor training")
