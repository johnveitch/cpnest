import pstats, cProfile

s = pstats.Stats("prof_sampler.prof")
s.strip_dirs().sort_stats("time").print_stats()

s = pstats.Stats("prof_nested_sampling.prof")
s.strip_dirs().sort_stats("cumulative").print_stats()
