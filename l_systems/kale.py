import bpy
import lsystem.exec
import random
import sys
import os

from datetime import datetime

def generate_lsystem():
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
    lSystem.name = "LSystem_Plant"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)

    from scripts.generate_mesh import generate_mesh

    generate_mesh(generate_lsystem, exclude_objects_with_collection="lsystem")

if __name__ == "__main__":
    main()