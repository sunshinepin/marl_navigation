import xuance
runner = xuance.get_runner(method=["iddpg", "maddpg"],
                           env='mpe',
                           env_id='simple_push_v3',
                           is_test=False)
runner.run()

