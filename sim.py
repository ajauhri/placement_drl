from __future__ import division

class Sim:
    def __init__(self, n_lng_grids):
        self.hours_per_day = 24
        self.pickup_maps = {}
        self.dropoff_maps = {}
        self.n_lng_grids = n_lng_grids

    def _update_map(self, src_t, src, dst_t, dst):
        self.dropoff_maps[src_t][src] -= 1
        self.pickup_maps[dst_t][dst] -= 1
 
    def get_hour_of_day(self, t):
        return int(t % self.hours_per_day)
   
    def add_maps(self, t, dropoff_map, pickup_map):
        dropoff_t = self.get_hour_of_day(t)
        pickup_t = self.get_hour_of_day(t+1)

        if dropoff_t in self.dropoff_maps:
            self.dropoff_maps[dropoff_t] += dropoff_map
        else:
            self.dropoff_maps[dropoff_t] = dropoff_map
        
        if pickup_t in self.pickup_maps:
            self.pickup_maps[pickup_t] += pickup_map
        else:
            self.pickup_maps[pickup_t] = pickup_map
    
    def get_next_node(self, curr_node, a):
        lng_idx = int(curr_node % self.n_lng_grids)
        lat_idx = int(curr_node / self.n_lng_grids)
        next_node = -1
        
        if a == -1:
            return curr_node

        if a == 1:
            if lat_idx - 1 < 0:
                next_node = [0, lng_idx]
            else:
                next_node = [lat_idx - 1, lng_idx]
        elif a == 3:
            if lat_idx + 1 >= self.n_lng_grids:
                next_node = [self.n_lng_grids - 1, lng_idx]
            else:
                next_node = [lat_idx + 1, lng_idx]

        elif a == 0:
            if lng_idx - 1 < 0:
                next_node = [lat_idx, 0]
            else:
                next_node = [lat_idx, lng_idx - 1]

        elif a == 2:
            if lng_idx + 1 >= self.n_lng_grids:
                next_node = [lat_idx, self.n_lng_grids - 1]
            else:
                next_node = [lat_idx, lng_idx + 1]

        return next_node[0] * self.n_lng_grids + next_node[1]

    def step(self, src, t, a):
        """
        s: state depicting the centroid of the dropoff grid, hour of day
        a: left (0), down (1), right (2), up (3), NOP
        """
        dropoff_t = self.get_hour_of_day(t)
        pickup_t = self.get_hour_of_day(t+1)
        dst = self.get_next_node(src, a)
        if self.dropoff_maps[dropoff_t][src] > 0 and \
                self.pickup_maps[pickup_t][dst] > 0:
                    self._update_map(dropoff_t, src, pickup_t, dst)
                    return 1, dst
        return 0, dst
