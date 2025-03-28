import bpy
import sys
import os
import lsystem.exec
import random

from datetime import datetime

def randnum(char, min=1, max=5):
    r = random.randint(min, max)    
    ret = ""
    for i in range(0, r):
        ret = ret + char
    return ret

def generate_lsystem():
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

    # Execute the L-system
    
    exec.exec(seed=datetime.now().timestamp(), 
        min_iterations=7, 
        angle=10, 
        length=0.5,
        radius=0.3,
        animate=False)

    # Rename lsystem
    lSystem = bpy.context.object
    lSystem.name = "LSystem_Plant"
    

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)

    from scripts.generate_mesh import generate_mesh
    
    generate_mesh(generate_lsystem, exclude_objects_with_collection="lsystem")

if __name__ == "__main__":
    main()