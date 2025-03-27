import lsystem.exec
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

if __name__ == "__main__":
    mint_l_system()