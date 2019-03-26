

class DisJoinSet(object):
    def __init__(self):
        self.groups = {}
        self.cnt = -1

    def append(self, items):
        '''
            automatically applly items array containg all item which is belong to the same group
            into a new group or into an exiting group
        '''
        if len(items) == 0:
            return
        mark = self.cnt + 1
        for item in items:
            if item in self.groups:
                mark = min(self.groups[item], mark)
        if mark == self.cnt + 1:
            self.cnt += 1
        for item in items:
            if item not in self.groups:
                self.groups[item] = mark
            elif self.groups[item] != mark:
                v = self.groups[item]
                keys = self.groups.keys()
                for key in keys:
                    if self.groups[key] == v:
                        self.groups[key] = mark

    def build_set(self):
        '''
            return  a set in which {key, value} means {name, groupname}
        '''
        groupall = {}
        values = self.groups.values()
        vis = [None for i in range(len(values))]
        for key in self.groups.keys():
            value = self.groups[key]
            if vis[value] is None:
                vis[value] = key
            groupall[key] = vis[value]
        return groupall
