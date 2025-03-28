import bpy
import sys
import os
import lsystem.exec
import random

from datetime import datetime

def generate_lsystem():
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

    exec.exec(seed=datetime.now().timestamp(), min_iterations=20, angle=25)
        
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