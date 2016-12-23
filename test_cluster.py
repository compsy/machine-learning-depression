from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Scattering part.
if rank == 0:
    data = [(i+1)**2 for i in range(size)]
else:
    data = None
data = comm.scatter(data, root=0)
assert data == (rank+1)**2

# Check if data is scattered accordingly.
print ("rank ", rank, "has data: ", data)

# Node dependent computations on data.
for i in range(size):
    if rank == i:
        data = data * rank

# Synchronization of the nodes.
comm.Barrier() 

# Gathering part.
data = comm.gather(data, root=0)
if rank == 0:
    print (data)
else:
    assert data is None 
quit()
