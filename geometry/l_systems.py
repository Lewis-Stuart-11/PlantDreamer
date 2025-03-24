#import lsystem.exec
import bpy
import random
from datetime import datetime

# Clear existing objects
def clearObjects():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

def randnum(char, min=1, max=5):
    r = random.randint(min, max)    
    ret = ""
    for i in range(0, r):
        ret = ret + char
    return ret

def bean_l_system():
    exec = lsystem.exec.Exec()

    random.seed(datetime.now().timestamp())
    
    slash = randnum("/")
    up = randnum("^")
    amp = randnum("&")
    
    exec.set_axiom("p(subsurf)/(rand(0, 360))I(5)aa(5)")
    exec.add_rule("aa(t)", "[&(70)B]/(rand(100, 160))I(10)^^^aa(sub(t,1))", "gt(t,0)") #if (t>0) start a branch, roll right 137.5, pitch down 70 B, I(10)aa(t-1)
    exec.add_rule("aa(t)", f"[&(70)B]/(rand(100, 160))I(10)aa(sub(t,1)){amp}Y{up}{slash}\Y^^^//\\\\\\\\Y", "eq(t,0)") #if (t>0) start a branch, roll right 137.5, pitch down 70 B, I(10)aa(t-1)
    #exec.add_rule("aa(t)", "[&(70)B]/(137.5)I(10)A", "eq(t,0)") #if t==0 start a branch, roll right 137.5, pitch down 70 B, A
    
    exec.add_rule("I(t)", "FI(sub(t,1))", "gt(t,0)") #if t > 0, Move branch forward, I(t-1)
    #exec.add_rule("I(t)", "F", "eq(t,0)") #if t==0, move forward
    
    exec.add_rule("B", "Fbb(0)", "gt(t,0)") #if t > 0, Move branch forward bb(2)
    
    exec.add_rule("bb(b)", "F&bb(sub(b,1))", "gt(b,0)") #if b > 0, Move branch forward, pitch down, bb(b-1)
    exec.add_rule("bb(b)", f"F^^^{up}X/////X", "eq(b,0)") #if b == 0, Move branch forward, pitch up place a leaf, roll right, place a leaf
    
    exec.add_rule("X", "~(leaf)")
    exec.add_rule("Y", "~(small_leaf)")
    
    exec.exec(seed=datetime.now().timestamp(), 
        min_iterations=7, 
        angle=10, 
        length=0.5,
        radius=0.3,
        animate=False)

    lSystem = bpy.context.object
    lSystem.name = "bean"

def kale_l_system():
    exec = lsystem.exec.Exec()
    
    exec.define("a0", "85")
    exec.define("a1", "10")
    exec.define("d1", "135")
    exec.define("d2", "145")
    exec.define("al", "0.6")
    exec.define("dl", "0.3")
    exec.define("bl", "0.5")
    exec.define("cl", "0.4")
    exec.define("wr", "0.85")
    
    random.seed(datetime.now().timestamp())
    exec.set_axiom("p(curve) [A(5,0.1)]")
    
    my_num = random.randint(0,60)
    
    exec.add_rule("A(t,w)", f"¤(w) /(rand(0,360)) $ F(0.6) [ &(rand(90,100)) /(rand(-45,45)) +(rand(-30,30)) B(bl,mul(w,wr)) ] -(4) /(rand(-45,45)) D(sub(t,1),mul(w,wr))")
    
    exec.add_rule("B(l,w)", f"¤(w) ^ F(l) L - (rand(20,40))/(rand(-45,45))")
        
    exec.add_rule("L", "F(0.05) ~(kleaf)")
        
    exec.add_rule("D(t,w)", "¤(w) F(0.4) [&(rand(90,100))/(rand(-45,45))] + (5)/(rand(-45,45)) A(sub(t,1),mul(w,wr))")
    
    exec.exec(seed=datetime.now().timestamp(), min_iterations=10, instances=10)
        
    # Rename lsystem
    lSystem = bpy.context.object
    lSystem.name = "kale"

def mint_l_system():
    exec = lsystem.exec.Exec()
    
    axiom_string = "p(curve) ¤(0.05) /(rand(0, 360)) [F(1) A(4)] /(120) [+(90)F(1.0)-(90)F(1) A(4)] /(120) [^(90)F(1.0)&(90)F(1) A(4)]"
    if random.random() < 0.75:
        axiom_string = axiom_string + "/(75) [-(90)F(0.75)+(85)F(1) A(4)]"
    if random.random() < 0.75:
        axiom_string = axiom_string + "/(75) [&(90)F(0.75)^(85)F(1) A(4)]"
    if random.random() < 0.75:
        axiom_string = axiom_string + "/(75) [-(90)F(1.5)+(85)F(0.5) A(3)]"
    if random.random() < 0.75:
        axiom_string = axiom_string + "/(75) [&(90)F(1.5)^(85)F(0.5) A(3)]"   
         
    #exec.set_axiom("p(curve) +(10)^ [A(8,0.1,rand(0,3))] -/^ F(0.4) \& [A(8,0.1,rand(0,3))] +/^ F(0.4) \& [A(8,0.1,rand(0,3))]")
    exec.set_axiom(axiom_string)
    
    exec.add_rule("A(t)", "F(1) B(t)")
    
    exec.add_rule("B(t)", "F(0.5) /(130) ^(rand(17,22)) [C(sub(t,1))]")
    
    exec.add_rule("C(t)", "F(0.5) &(rand(10,15)) B(t)[X]", "lt(t, 0)")
    exec.add_rule("C(t)", "F(0.5) &(rand(10,15)) B(t)[X]", "lt(t, 2)")
    exec.add_rule("C(t)", "F(0.5) &(rand(10,15)) B(t)[Z]", "gteq(t, 0)")
    
    exec.add_rule("X", "F(0.1) ~(mleaf)")
    
    exec.add_rule("Y", "F(0.1) ~(smleaf)")
    
    exec.add_rule("Z", "F(0.1) ~(tmleaf)")
    
    #exec.exec(seed=datetime.now().timestamp(), min_iterations=25)
    #exec.exec(seed=datetime.now().timestamp(), min_iterations=10, angle=30)
    #exec.exec(seed=6, min_iterations=25, angle=25)
    exec.exec(seed=datetime.now().timestamp(), min_iterations=20, angle=25)
        
    # Rename lsystem
    lSystem = bpy.context.object
    lSystem.name = "mint"


def get_l_system(plant_species):
    plant_type_str = plant_type_str.lower().strip()

    if plant_type_str == "bean":
        return bean_l_system()
    elif plant_type_str == "kale":
        return kale_l_system()
    elif plant_type_str == "mint":
        return mint_l_system()

