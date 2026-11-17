# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .graph import *


def delete(x, y, H, C, legacy=False):
    """
    delete the edge between x and y, and for each h in H:
    (1) delete the previously undirected edge between x and y;
    (2) directing any previously undirected edge between x and h in H as x->h.

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    H: the neighbors of y that are adjacent to x
    C: numpy.array
        CPDAG
    legacy: bool
        if True, use gCastle's original third step, which orients h -> y
        again instead of x -> h

    Returns
    -------
    out: numpy.array
        new C
    """

    # first operate
    C[x, y] = 0
    C[y, x] = 0

    # second operate: every h in H becomes a child of y
    if len(H) != 0:
        C[list(H), y] = 0

    # third operate: every previously undirected x - h becomes x -> h
    x_neighbor = neighbors(x, C)
    H_and_neighbor = H & x_neighbor
    if len(H_and_neighbor) != 0:
        if legacy:
            C[list(H_and_neighbor), y] = 0
        else:
            C[list(H_and_neighbor), x] = 0

    return C


def delete_validity(x, y, H, C):
    """
    check whether a delete operator is valid

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    H: the neighbors of y that are adjacent to x
    C: numpy.array
        CPDAG

    Returns
    -------
    out: bool
        if True denotes the operator is valid, else False.
    """

    na_yx = neighbors(y, C) & adjacent(x, C)
    na_yx_h = na_yx - H

    # only one condition
    condition = is_clique(na_yx_h, C)

    return condition