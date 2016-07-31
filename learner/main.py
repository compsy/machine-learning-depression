from driver import Driver

if __name__ == '__main__':
    print('Main called, lets do this guys!')
    Driver(verbosity=0,
           hpc=True,
           hpc_log=True,
           polynomial_features=False,
           normalize=False,
           scale=True,
           classification=True,
           force_no_caching=True)
