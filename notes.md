- Started with PD controller

- Inverse Kinematics
- Remove Arms and ankles to make spheres
- First make it work without wrist and ankles
- First confiuration was belly down but bely up better
- Best Approach whic I am gonna explore -> IK for joints RL for foot

## 🧠 Research-Grade Architecture

### Overview
This system separates **high-level decision making** from **low-level control**:

- **Reinforcement Learning (RL)** → decides *what to do*
- **Inverse Kinematics (IK)** → decides *how to do it*

---

### 🧠 RL Responsibilities
RL handles:
- **Coordination** → synchronizing all limbs  
- **Gait timing** → when each limb moves or stays in contact  
- **Balance** → maintaining stability under dynamics  

---

### 🦾 IK Responsibilities
IK handles:
- **Geometry** → computing joint angles for desired end-effector positions  
- **Joint constraints** → respecting limits and kinematic structure  

---

### 🔁 Pipeline


---

### ⚡ Key Insight

> RL learns **strategy**, IK handles **execution**.

This separation:
- simplifies learning
- improves stability
- makes the system modular and scalable
