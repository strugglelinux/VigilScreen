"""仅关联位置以防重复提示，身份与视线始终使用当前帧重新判断。"""
from dataclasses import dataclass
import math


@dataclass
class Track:
    box: tuple
    last_seen: float
    gate: object
    missing: int = 0


def overlap(one,two):
    x,y,w,h = one
    a,b,c,d = two
    intersection = max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
    union = w*h+c*d-intersection
    return intersection/union if union>0 else 0


class PersonHits:
    def __init__(self,gate_factory):
        self.gate_factory = gate_factory
        self.tracks = {}
        self.next_id = 1

    def update(self,rows,now):
        self.tracks = {key:track for key,track in self.tracks.items() if now-track.last_seen<=2}
        candidates = []
        for index,row in enumerate(rows):
            box = tuple(row[:4])
            for key,track in self.tracks.items():
                iou = overlap(box,track.box)
                distance = math.dist((box[0]+box[2]/2,box[1]+box[3]/2),
                                     (track.box[0]+track.box[2]/2,track.box[1]+track.box[3]/2))
                size = max(box[2],box[3],track.box[2],track.box[3],1)
                if iou>=.2 or distance/size<.45:
                    candidates.append((iou-distance/size,index,key))
        assignments,used = {},set()
        for score,index,key in sorted(candidates,reverse=True):
            if index not in assignments and key not in used:
                assignments[index] = key
                used.add(key)
        for key in list(self.tracks):
            if key not in used:
                track = self.tracks[key]
                track.gate.update(False,now)
                track.missing += 1
                if track.missing>=3:
                    del self.tracks[key]
        hits = []
        ids = []
        for index,row in enumerate(rows):
            key = assignments.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = Track(tuple(row[:4]),now,self.gate_factory())
            track = self.tracks[key]
            track.box,track.last_seen,track.missing = tuple(row[:4]),now,0
            ids.append(key)
            if track.gate.update(bool(row[6]),now):
                hits.append(key)
        return hits,ids
