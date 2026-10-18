import numpy as np
M=0.5
m=0.2
l=0.3
g=9.81
A = np.array([[0,1,0,0],
             [0,0,-((m*g)/M),0],
             [0,0,0,1],
             [0,0,((M+m)*g)/(M*l),0]],dtype=float)
B = np.array([[0],
             [1/M],
             [0],
             [-1/(M*l)]],dtype=float)


C = B.copy()
AB = B.copy()

#C = [B, A@B, A@A@B, A@A@A@B]
for i in range(1,4):
    AB = A@AB
    C = np.hstack((C,AB))

print(f"ctrb:{C}")
print(f"rank of ctrb: {np.linalg.matrix_rank(C)}")
print(f"상태변수의 수: {A.shape[0]}")


if np.linalg.matrix_rank(C) == A.shape[0]:
    print("가제어성: controllable")
else:
    print("가제어성: uncontrollable")

D = np.array([
    [1,0,0,0],
    [0,0,1,0]
],dtype=float)


O = D.copy()
CA = D.copy()

for i in range(1,A.shape[0]):
    CA = CA@A
    O = np.vstack((O,CA))
print(f"ovsb: {O}")
rank = np.linalg.matrix_rank(O)
print(f"rank: {rank}")
print(f"상태변수: {A.shape[0]}")
if rank == A.shape[0]:
    print("가관측성: observable")
else:
    print("가관측성: unobservable")