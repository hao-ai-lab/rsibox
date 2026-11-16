"""The TP group of the toy: rank 0 is the hub over one duplex pipe per peer (SGLang uses the
gloo `cpu_group` for objects and NCCL for tensors; both block until every rank arrives)."""


class TpGroup:
    def __init__(self, rank, size, conns):
        self.rank, self.size = rank, size
        self.conns = conns          # rank 0: {peer rank: conn}; a peer: {0: conn}

    def broadcast_pyobj(self, obj):
        if self.size == 1:
            return obj
        if self.rank == 0:
            for peer in range(1, self.size):
                self.conns[peer].send(obj)
            return obj
        return self.conns[0].recv()

    def all_gather(self, shard):
        """Every rank's shard, in rank order, on every rank."""
        if self.size == 1:
            return [shard]
        if self.rank == 0:
            shards = [shard] + [self.conns[peer].recv() for peer in range(1, self.size)]
            for peer in range(1, self.size):
                self.conns[peer].send(shards)
            return shards
        self.conns[0].send(shard)
        return self.conns[0].recv()

    def barrier(self):
        self.all_gather(None)
