from driver import Driver

if __name__ == '__main__':
    Driver(verbosity=0,
           hpc=True,
           hpc_log=True,
           polynomial_features=True,
           normalize=False,
           scale=True,
           classification=True,
           force_no_caching=False)
