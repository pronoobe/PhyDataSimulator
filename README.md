# PhyDataSimulator
一个基于python的、便于使用的物理引擎
使用方法：
a = World(d_time=0.001, Gravity=9.8, atom_height=100000, default_height=0, temperature=20, get_data_tick=0.1,
              floor=0,
              friction=1) #设置基础参数
    a.SetBasicSolid(solid_type="C", init_pos=[30, 110], speed=10, angel=90, solid_round=8, name="222", mass=20)#放置物体
    a.SetBasicSolid(solid_type="C", init_pos=[30, 140], speed=22, angel=-90, solid_round=10, name="2222", mass=20)
    a.SetBasicSolid(solid_type="C", init_pos=[47, 133], speed=22, angel=-30, solid_round=12, name="2s222", mass=20)
    a.SetField(field_type="Wind", field_name="field1", filed_data=Vector(r=200, o=60)) #放置场
