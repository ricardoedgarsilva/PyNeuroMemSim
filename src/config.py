config = {
    "simulation":{
        "save"  : rf"C:\Users\ricar\Desktop\results",
        "dataset"   : "satdap",
        # "geometry": [[36,18],[18,6],[6,3]],
        "geometry": [[36,3]],
        "test_size" : 0.1,
        "timestep"  : 1e-8,
        "freq"      : 1e8,
        "precision" : 10,
        "epochs"    : 50,
        "learning_rate" : 6e-4
    },
    "opamp":{
        "power" : 1,
        "noninverting" : 0 # Crashing LTSpice solver if > 0.2
    },
    "resistor":{
        "A" : 50,
        "B" : 1e3,
        "C" : 1e3
    },
    "memristor":{
        # 0: Biolek, 1: Yakopcic
        "model":  0,
        "parameters": [
            {
            "Ron"   : 800,
            "Roff"  : 2000,
            "D"     : 0.2212, 
            "uv"    : 0.0108, 
            "p"     : 4.64
            },
            {
            "a1" : 0.32,
            "a2" : 0.32,
            "b" :  0.00032,
            "Vp" : 2.97,
            "Vn" : 2.17,
            "Ap" : 4000,
            "An" : 4000,
            "xp" : 0.3,
            "xn" : 0.52,
            "alphap" : 0.82,
            "alphan" : 1,
            "xo" : 0.01,
            "eta" : 1
            }
        ],
        "subcircuits" : 
        [
'''* Biolek Memristor Model Subcircuit
.SUBCKT memristor TE BE XSV
.params Ron={Ron} Roff={Roff} D={D} uv={uv} p={p} xo={{xo}}

* Biolek Window Function
.func f(V1,I1)={{1-pow((V1-stp(-I1)),(2*p))}}

* Memristor I-V Relationship
.func IVRel(V1,V2) = V1/(Ron*V2 + Roff*(1-V2))

* Circuit to determine state variable
Gx 0 XSV value={{I(Gmem)*Ron*uv*f(V(XSV,0),I(Gmem))/pow(D,2)}}
Cx XSV 0 {{1}}
.ic V(XSV) = xo

* Current source representing memristor
Gmem TE BE value={{IVRel(V(TE,BE),V(XSV,0))}}

.ENDS memristor'''

,

'''* Yakopcic Memristor Model
.subckt memristor TE BE XSV 
.params a1={a1} a2={a2} b={b} Vp={Vp} Vn={Vn} Ap={Ap} 
+An={An} xp={xp} xn={xn} alphap={alphap} alphan={alphan} eta={eta} xo={{xo}}

* Multiplicitive functions to ensure zero state
* variable motion at memristor boundaries
.func wp(V) = xp/(1-xp) - V/(1-xp) + 1
.func wn(V) = V/(1-xn)

* Function G(V(t)) - Describes the device threshold
.func G(V) = IF(V <= Vp, IF(V >= -Vn, 0, -An*(exp(-
+V)-exp(Vn))), Ap*(exp(V)-exp(Vp)))

* Function F(V(t),x(t)) - Describes the SV motion 
.func F(V1,V2) = IF(eta*V1 >= 0, IF(V2 >= xp, exp(-
+alphap*(V2-xp))*wp(V2) ,1), IF(V2 <= (1-xn), 
+exp(alphan*(V2+xn-1))*wn(V2) ,1))

* IV Response - Hyperbolic sine due to MIM structure
.func IVRel(V1,V2) = IF(V1 >= 0, a1*V2*sinh(b*V1),
+a2*V2*sinh(b*V1) )

* Circuit to determine state variable
* dx/dt = F(V(t),x(t))*G(V(t))
Cx XSV 0 {{1}}
.ic V(XSV) = xo
Gx 0 XSV value={{eta*F(V(TE,BE),V(XSV,0))*G(V(TE,BE))}}
* Current source for memristor IV response
Gm TE BE value = {{IVRel(V(TE,BE),V(XSV,0))}}

.ends memristor'''

,

'''* Joglekar Memristor Model
.SUBCKT memristor TE BE XSV 
.params Ron={Ron} Roff={Roff} D={D} uv={uv} p={p} xo={{xo}}

* Joglekar Window Function
.func f(V1) = 1-pow((2*V1-1),(2*p))

* Memristor I-V Relationship
.func IVRel(V1,V2) = V1/(Ron*V2 + Roff*(1-V2))

* Circuit to determine state variable
Gx 0 XSV value={{ I(Gmem)*Ron*uv*f(V(XSV,0))/pow(D,2) }}
Cx XSV 0 {{1}}
.ic V(XSV) = xo

* Current source representing memristor
Gmem TE BE value={{IVRel(V(TE,BE),V(XSV,0))}}

.ENDS memristor'''
        ]
    }
}



