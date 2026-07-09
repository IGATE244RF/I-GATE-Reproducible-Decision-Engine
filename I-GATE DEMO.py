# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:03:20 2026

@author: Elyar
"""

# ============================================================
# IGATE RESEARCH ENGINE
#
# Version : 1.0 Research Release
#
# Author : I-GATE Research
#
# Phase 1 : Enterprise Data Layer
#
# ============================================================

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx

from scipy.integrate import solve_ivp

# ============================================================
# Logger
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(levelname)s | %(message)s"

)

logger=logging.getLogger("IGATE")


# ============================================================
# Configuration
# ============================================================

class Config:

    RANDOM_SEED=42

    DATA_FOLDER="data"

    RESULT_FOLDER="results"

    GRAPH_NAME="IGATE_MCKG"

    VERSION="1.0"

    DEFAULT_CONFIDENCE=1.0

    DEFAULT_WEIGHT=1.0


np.random.seed(Config.RANDOM_SEED)

# ============================================================
# Exceptions
# ============================================================

class IGATEException(Exception):
    pass

class DatasetException(IGATEException):
    pass

class GraphException(IGATEException):
    pass

class StateException(IGATEException):
    pass


# ============================================================
# Enterprise Dataset
# ============================================================

@dataclass

class EnterpriseDataset:

    name:str

    dataframe:pd.DataFrame

    source:str

    metadata:Dict[str,Any]=field(default_factory=dict)

    def rows(self):

        return len(self.dataframe)

    def columns(self):

        return list(self.dataframe.columns)

    def shape(self):

        return self.dataframe.shape

    def summary(self):

        return {

            "rows":self.rows(),

            "columns":self.columns(),

            "shape":self.shape(),

            "source":self.source

        }
# ============================================================
# Data Loader
# ============================================================

class DataLoader:

    def __init__(self):

        self.datasets={}

    def register(self,dataset):

        self.datasets[dataset.name]=dataset

    def names(self):

        return list(self.datasets.keys())

    def get(self,name):

        if name not in self.datasets:

            raise DatasetException(name)

        return self.datasets[name]

    def summary(self):

        for ds in self.datasets.values():

            logger.info(ds.summary())
class CSVLoader:

    def load(self,path):

        df=pd.read_csv(path)

        return EnterpriseDataset(

            name=Path(path).stem,

            dataframe=df,

            source="CSV"

        )
class ExcelLoader:

    def load(self,path,sheet_name=0):

        df=pd.read_excel(

            path,

            sheet_name=sheet_name

        )

        return EnterpriseDataset(

            name=Path(path).stem,

            dataframe=df,

            source="Excel"

        )
class JSONLoader:

    def load(self,path):

        with open(

            path,

            "r",

            encoding="utf8"

        ) as f:

            obj=json.load(f)

        df=pd.DataFrame(obj)

        return EnterpriseDataset(

            name=Path(path).stem,

            dataframe=df,

            source="JSON"

        )
class DataValidator:

    @staticmethod

    def check_missing(dataset):

        return dataset.dataframe.isnull().sum()

    @staticmethod

    def check_duplicate(dataset):

        return dataset.dataframe.duplicated().sum()

    @staticmethod

    def validate(dataset):

        report={}

        report["rows"]=dataset.rows()

        report["columns"]=len(dataset.columns())

        report["missing"]=DataValidator.check_missing(dataset)

        report["duplicates"]=DataValidator.check_duplicate(dataset)

        return report
if __name__=="__main__":

    logger.info("IGATE Research Engine Started")

    loader=DataLoader()

    print("Phase 1 Ready")
# ============================================================
# Phase 2
# Mathematical Computational Knowledge Graph (MCKG)
# ============================================================

from enum import Enum


class NodeType(Enum):

    PRODUCT="Product"

    REQUIREMENT="Requirement"

    GATE="Gate"

    STATE="State"

    RISK="Risk"

    KPI="KPI"

    SCENARIO="Scenario"

    AGENT="Agent"

    DOCUMENT="Document"

    PARAMETER="Parameter"

    CONTROL="Control"

    UNKNOWN="Unknown"


class EdgeType(Enum):

    INFLUENCE="Influence"

    DEPENDENCY="Dependency"

    CONTROL="Control"

    KNOWLEDGE="Knowledge"

    GAME="Game"

    FLOW="Flow"

    SEMANTIC="Semantic"

@dataclass

class MCKGNode:

    uid:str

    name:str

    node_type:NodeType

    attributes:dict=field(default_factory=dict)

    state_index:int=-1

    value:float=0.0

    confidence:float=1.0

    dynamic:bool=True

    history:list=field(default_factory=list)

    equation=None

    def set(self,key,value):

        self.attributes[key]=value

    def get(self,key,default=None):

        return self.attributes.get(key,default)

    def update(self,value):

        self.value=value

        self.history.append(value)

    def __repr__(self):

        return f"{self.node_type.value}:{self.name}"

@dataclass

class MCKGEdge:

    source:str

    target:str

    edge_type:EdgeType

    weight:float=1.0

    delay:float=0.0

    confidence:float=1.0

    sign:int=1

    metadata:dict=field(default_factory=dict)

    def __repr__(self):

        return f"{self.source}->{self.target}"

class MCKG:

    def __init__(self,name="IGATE"):

        self.name=name

        self.graph=nx.MultiDiGraph()

        self.nodes={}

        self.edges=[]

    # -----------------------------------

    def add_node(self,node):

        self.nodes[node.uid]=node

        self.graph.add_node(

            node.uid,

            object=node

        )

    # -----------------------------------

    def add_edge(self,edge):

        self.edges.append(edge)

        self.graph.add_edge(

            edge.source,

            edge.target,

            object=edge,

            weight=edge.weight,

            relation=edge.edge_type.value

        )

    # -----------------------------------

    def get_node(self,uid):

        return self.nodes.get(uid)

    # -----------------------------------

    def find(self,name):

        for n in self.nodes.values():

            if n.name==name:

                return n

        return None

    # -----------------------------------

    def number_of_nodes(self):

        return len(self.nodes)

    # -----------------------------------

    def number_of_edges(self):

        return len(self.edges)

    # -----------------------------------

    def summary(self):

        print()

        print("========== MCKG ==========")

        print("Name :",self.name)

        print("Nodes :",self.number_of_nodes())

        print("Edges :",self.number_of_edges())

        print("==========================")

class Ontology:

    def __init__(self):

        self.dictionary={}

    def register(

        self,

        keyword,

        node_type

    ):

        self.dictionary[keyword.lower()]=node_type

    def classify(

        self,

        text

    ):

        t=text.lower()

        for k,v in self.dictionary.items():

            if k in t:

                return v

        return NodeType.UNKNOWN

class SemanticMapper:

    def __init__(

        self,

        ontology

    ):

        self.ontology=ontology

    def dataframe_to_graph(

        self,

        dataset

    ):

        graph=MCKG(dataset.name)

        for col in dataset.columns():

            t=self.ontology.classify(col)

            node=MCKGNode(

                uid=str(uuid.uuid4()),

                name=col,

                node_type=t

            )

            graph.add_node(node)

        return graph

ontology=Ontology()

ontology.register(

    "cost",

    NodeType.KPI

)

ontology.register(

    "quality",

    NodeType.STATE

)

ontology.register(

    "risk",

    NodeType.RISK

)

ontology.register(

    "time",

    NodeType.KPI

)

ontology.register(

    "gate",

    NodeType.GATE

)

ontology.register(

    "scenario",

    NodeType.SCENARIO
)

ontology.register(

    "agent",

    NodeType.AGENT
)

if __name__=="__main__":

    df=pd.DataFrame(

        {

            "Cost":[1,2],

            "Quality":[0.7,0.8],

            "Risk":[0.4,0.3],

            "Gate":[1,2]

        }

    )

    ds=EnterpriseDataset(

        "Demo",

        df,

        "Memory"

    )

    mapper=SemanticMapper(

        ontology

    )

    kg=mapper.dataframe_to_graph(

        ds

    )

    kg.summary()

# ============================================================
# Phase 3
# State Space Engine
# MCKG  --->  State Vector
# ============================================================

class StateVariable:
    """
    One Dynamic State
    """

    def __init__(
            self,
            index,
            name,
            value=0.0):

        self.index=index
        self.name=name
        self.value=float(value)

        self.initial=float(value)

        self.history=[]

    def set(self,value):

        self.value=float(value)

        self.history.append(self.value)

    def reset(self):

        self.value=self.initial

        self.history=[]

    def __repr__(self):

        return f"x[{self.index}]={self.name}:{self.value:.4f}"

class StateVector:

    """
    x(t)
    """

    def __init__(self):

        self.states=[]

        self.lookup={}

    # ------------------------------

    def add(

        self,

        state

    ):

        self.lookup[state.name]=len(self.states)

        state.index=len(self.states)

        self.states.append(state)

    # ------------------------------

    def size(self):

        return len(self.states)

    # ------------------------------

    def values(self):

        return np.array(

            [

                s.value

                for s in self.states

            ],

            dtype=float

        )

    # ------------------------------

    def set_values(

        self,

        x

    ):

        for i,v in enumerate(x):

            self.states[i].set(v)

    # ------------------------------

    def names(self):

        return [

            s.name

            for s in self.states

        ]

    # ------------------------------

    def summary(self):

        print()

        print("State Vector")

        print("----------------")

        for s in self.states:

            print(s)

class StateExtractor:

    """
    Graph

      ↓

    State Vector

    """

    def __init__(

        self,

        graph

    ):

        self.graph=graph

    def build(self):

        vector=StateVector()

        index=0

        for node in self.graph.nodes.values():

            if node.dynamic:

                node.state_index=index

                s=StateVariable(

                    index,

                    node.name,

                    node.value

                )

                vector.add(s)

                index+=1

        return vector

class StateMapper:

    def __init__(

        self,

        vector

    ):

        self.vector=vector

    def node_to_state(

        self,

        name

    ):

        return self.vector.lookup[name]

    def state_to_node(

        self,

        index

    ):

        return self.vector.states[index]

class StateInitializer:

    def initialize(

        self,

        vector,

        dataframe

    ):

        cols=dataframe.columns

        row=dataframe.iloc[0]

        for s in vector.states:

            if s.name in cols:

                try:

                    s.value=float(

                        row[s.name]

                    )

                except:

                    s.value=0.0

        return vector

if __name__=="__main__":

    extractor=StateExtractor(

        kg

    )

    x=extractor.build()

    initializer=StateInitializer()

    x=initializer.initialize(

        x,

        ds.dataframe

    )

    x.summary()

    print()

    print("x(t) =")

    print(

        x.values()

    )

# ============================================================
# Phase 4
# Dynamic System Engine
# ============================================================

class DynamicFunction:
    """
    One Dynamic Equation

    dx_i/dt = f_i(x,u)
    """

    def __init__(

            self,

            state_name,

            function=None

    ):

        self.state_name=state_name

        self.function=function

    def evaluate(

            self,

            t,

            x,

            u

    ):

        if self.function is None:

            return 0.0

        return self.function(

            t,

            x,

            u

        )

class DynamicSystem:

    """
    Nonlinear Dynamic System

    dx/dt=f(x,u)

    """

    def __init__(self):

        self.functions=[]

        self.state_names=[]

    def add_equation(

            self,

            state_name,

            function

    ):

        self.state_names.append(

            state_name

        )

        self.functions.append(

            DynamicFunction(

                state_name,

                function

            )

        )

    def derivative(

            self,

            t,

            x,

            u

    ):

        dx=[]

        for f in self.functions:

            dx.append(

                f.evaluate(

                    t,

                    x,

                    u

                )

            )

        return np.asarray(

            dx,

            dtype=float

        )

class DynamicSimulator:

    def __init__(

            self,

            system

    ):

        self.system=system

    def simulate(

            self,

            x0,

            u,

            t0=0,

            tf=20,

            points=200

    ):

        t=np.linspace(

            t0,

            tf,

            points

        )

        sol=solve_ivp(

            lambda time,state:

            self.system.derivative(

                time,

                state,

                u

            ),

            [

                t0,

                tf

            ],

            x0,

            t_eval=t

        )

        return sol

system=DynamicSystem()

system.add_equation(

    "Cost",

    lambda t,x,u:

    -0.20*x[0]

    +0.40*x[2]

    +0.80*u[0]

)

system.add_equation(

    "Quality",

    lambda t,x,u:

    -0.10*x[1]

    +0.60*x[3]

    +0.70*u[1]

)

system.add_equation(

    "Risk",

    lambda t,x,u:

    -0.30*x[2]

    -0.50*x[1]

    -0.60*u[1]

)

system.add_equation(

    "Gate",

    lambda t,x,u:

    -0.05*x[3]

)

x0=np.array(

    [

        100,

        0.7,

        0.4,

        1.0

    ]

)

u=np.array(

    [

        10,

        2

    ]

)

sim=DynamicSimulator(

    system

)

result=sim.simulate(

    x0,

    u

)

print(result.y[:,-1])

# ============================================================
# Phase 5
# Automatic Equation Generator
# ============================================================

class EquationGenerator:

    """
    Build Dynamic Equations

    from MCKG

    """

    def __init__(

            self,

            graph,

            state_vector

    ):

        self.graph=graph

        self.vector=state_vector

        self.state_lookup={}

        for i,s in enumerate(

                self.vector.states

        ):

            self.state_lookup[s.name]=i

    # -------------------------------------------------

    def build(self):

        equations=[]

        controls=[]

        names=[]

        n=self.vector.size()

        for state in self.vector.states:

            idx=self.state_lookup[state.name]

            def create_equation(i):

                def equation(t,x,u):

                    value=0.0

                    node=self.graph.find(

                        self.vector.states[i].name

                    )

                    if node is None:

                        return 0.0

                    uid=node.uid

                    for edge in self.graph.edges:

                        if edge.target!=uid:

                            continue

                        source=self.graph.get_node(

                            edge.source

                        )

                        if source is None:

                            continue

                        if source.name in self.state_lookup:

                            j=self.state_lookup[source.name]

                            value+=edge.weight*x[j]

                    value-=0.10*x[i]

                    return value

                return equation

            equations.append(

                create_equation(idx)

            )

            names.append(

                state.name

            )

        return names,equations
class GraphDynamicSystem:

    def __init__(

            self,

            graph,

            vector

    ):

        self.system=DynamicSystem()

        builder=EquationGenerator(

            graph,

            vector

        )

        names,eqs=builder.build()

        for n,e in zip(

                names,

                eqs

        ):

            self.system.add_equation(

                n,

                e

            )

graph_system=GraphDynamicSystem(

    kg,

    x

)

solver=DynamicSimulator(

    graph_system.system

)

u=np.zeros(3)

result=solver.simulate(

    x.values(),

    u

)

print(result.y[:,-1])

# ============================================================
# Phase 6
# Jacobian Builder
# ============================================================

class JacobianBuilder:
    """
    Numerical Jacobian Builder

    A = df/dx
    B = df/du
    """

    def __init__(self,
                 system,
                 eps=1e-6):

        self.system = system
        self.eps = eps

    # --------------------------------------------------

    def compute_A(self,
                  x,
                  u):

        n = len(x)

        A = np.zeros((n, n))

        f0 = self.system.derivative(
            0.0,
            x,
            u
        )

        for j in range(n):

            xp = x.copy()

            xp[j] += self.eps

            fp = self.system.derivative(
                0.0,
                xp,
                u
            )

            A[:, j] = (fp - f0) / self.eps

        return A

    # --------------------------------------------------

    def compute_B(self,
                  x,
                  u):

        n = len(x)

        m = len(u)

        B = np.zeros((n, m))

        f0 = self.system.derivative(
            0.0,
            x,
            u
        )

        for j in range(m):

            up = u.copy()

            up[j] += self.eps

            fp = self.system.derivative(
                0.0,
                x,
                up
            )

            B[:, j] = (fp - f0) / self.eps

        return B

# ============================================================
# State Space Model
# ============================================================

class StateSpaceModel:

    def __init__(self,
                 A,
                 B):

        self.A = A
        self.B = B

    def derivative(self,
                   x,
                   u):

        return self.A @ x + self.B @ u

    def eigenvalues(self):

        return np.linalg.eigvals(self.A)

    def is_stable(self):

        eig = self.eigenvalues()

        return np.all(np.real(eig) < 0)

# ============================================================
# Test
# ============================================================

builder = JacobianBuilder(
    graph_system.system
)

A = builder.compute_A(
    x.values(),
    u
)

B = builder.compute_B(
    x.values(),
    u
)

print("\nA Matrix")
print(A)

print("\nB Matrix")
print(B)

ss = StateSpaceModel(
    A,
    B
)

print("\nEigenvalues")
print(
    ss.eigenvalues()
)

print(
    "\nStable :",
    ss.is_stable()
)

# ============================================================
# Phase 7
# Scenario Engine
# ============================================================

from copy import deepcopy


class Scenario:

    """
    One Scenario
    """

    def __init__(

            self,

            name

    ):

        self.name=name

        self.state_changes={}

        self.control_changes={}

        self.parameter_changes={}

        self.edge_changes=[]

    # ---------------------------------------

    def set_state(

            self,

            state,

            value

    ):

        self.state_changes[state]=value

    # ---------------------------------------

    def set_control(

            self,

            control,

            value

    ):

        self.control_changes[control]=value

    # ---------------------------------------

    def set_parameter(

            self,

            parameter,

            value

    ):

        self.parameter_changes[parameter]=value

class ScenarioEngine:

    def __init__(

            self,

            graph,

            state_vector

    ):

        self.graph=graph

        self.vector=state_vector

    # ---------------------------------------

    def apply(

            self,

            scenario

    ):

        new_vector=deepcopy(

            self.vector

        )

        for s in new_vector.states:

            if s.name in scenario.state_changes:

                s.value=scenario.state_changes[

                    s.name

                ]

        return new_vector

class ScenarioLibrary:

    def __init__(self):

        self.scenarios={}

    def add(

            self,

            scenario

    ):

        self.scenarios[scenario.name]=scenario

    def get(

            self,

            name

    ):

        return self.scenarios[name]

    def names(self):

        return list(

            self.scenarios.keys()

        )

library=ScenarioLibrary()

optimistic=Scenario(

    "Optimistic"

)

optimistic.set_state(

    "Quality",

    0.90

)

optimistic.set_state(

    "Risk",

    0.15

)

library.add(

    optimistic

)

# -----------------------------

pessimistic=Scenario(

    "Pessimistic"

)

pessimistic.set_state(

    "Quality",

    0.40

)

pessimistic.set_state(

    "Risk",

    0.75

)

library.add(

    pessimistic
)

engine=ScenarioEngine(

    kg,

    x

)

new_state=engine.apply(

    library.get(

        "Optimistic"

    )

)

print()

print("Scenario")

new_state.summary()

# ============================================================
# Phase 8
# Multi Agent Game Engine
# ============================================================

class Agent:

    """
    Intelligent Decision Agent
    """

    def __init__(
            self,
            name,
            controls):

        self.name=name

        self.controls=controls

        self.action=np.zeros(controls)

        self.weight=1.0

        self.utility_value=0.0

    def decide(self,state):

        """
        Initial Policy
        (Later replaced by Optimal Control)
        """

        self.action=np.random.rand(
            self.controls
        )

        return self.action

class UtilityFunction:

    """
    U(x,u)

    """

    def __init__(self):

        self.Q=None

        self.R=None

    def configure(

            self,

            Q,

            R

    ):

        self.Q=Q

        self.R=R

    def evaluate(

            self,

            x,

            u

    ):

        x=np.asarray(x)

        u=np.asarray(u)

        return (

            -(

                x.T@self.Q@x

            )

            -

            (

                u.T@self.R@u

            )

        )

class DynamicGame:

    """
    Dynamic Multi-Agent Game
    """

    def __init__(

            self,

            agents,

            utility

    ):

        self.agents=agents

        self.utility=utility

    def play(

            self,

            state

    ):

        actions=[]

        values=[]

        for agent in self.agents:

            u=agent.decide(

                state

            )

            J=self.utility.evaluate(

                state,

                u

            )

            agent.utility_value=J

            actions.append(u)

            values.append(J)

        return actions,values

PM=Agent(

    "Project Manager",

    2

)

Quality=Agent(

    "Quality Manager",

    2

)

Finance=Agent(

    "Finance Manager",

    2

)

agents=[

    PM,

    Quality,

    Finance

]

Q=np.eye(

    len(x.values())

)

R=np.eye(

    2

)

utility=UtilityFunction()

utility.configure(

    Q,

    R

)

game=DynamicGame(

    agents,

    utility

)

actions,values=game.play(

    x.values()

)

print()

for a,v in zip(

        agents,

        values

):

    print(

        a.name,

        v

    )

# ============================================================
# Phase 9
# Nash Equilibrium Solver
# ============================================================

class NashAgent(Agent):

    def __init__(
            self,
            name,
            controls):

        super().__init__(name,controls)

        self.lower=-1.0
        self.upper= 1.0

    # --------------------------------------------------

    def best_response(
            self,
            x,
            others):

        """
        Prototype Best Response

        در نسخه فعلی:
        پاسخ بهینه از روی وضعیت سیستم محاسبه می‌شود.

        در Release C
        از Hamiltonian استفاده خواهد شد.
        """

        action=np.zeros(self.controls)

        norm=np.linalg.norm(x)

        if norm<1e-12:
            norm=1.0

        value=1.0/(1.0+norm)

        action[:]=value

        action=np.clip(

            action,

            self.lower,

            self.upper

        )

        return action

class NashEquilibriumSolver:

    """
    Fixed Point Nash Solver
    """

    def __init__(
            self,
            agents):

        self.agents=agents

    # --------------------------------------------------

    def solve(

            self,

            state,

            max_iter=100,

            tolerance=1e-6

    ):

        actions=[

            np.zeros(a.controls)

            for a in self.agents

        ]

        for k in range(max_iter):

            error=0.0

            for i,agent in enumerate(self.agents):

                old=actions[i].copy()

                others=[

                    actions[j]

                    for j in range(len(actions))

                    if j!=i

                ]

                new=agent.best_response(

                    state,

                    others

                )

                actions[i]=new

                error=max(

                    error,

                    np.linalg.norm(

                        new-old

                    )

                )

            if error<tolerance:

                break

        return {

            "iterations":k+1,

            "actions":actions,

            "converged":error<tolerance

        }

agents=[

    NashAgent(

        "Project Manager",

        2

    ),

    NashAgent(

        "Quality",

        2

    ),

    NashAgent(

        "Finance",

        2

    )

]

solver=NashEquilibriumSolver(

    agents

)

result=solver.solve(

    x.values()

)

print()

print(result)

# ============================================================
# Phase 10
# Hamiltonian Engine
# ============================================================

class RunningCost:

    """
    L(x,u)

    """

    def __init__(

            self,

            Q,

            R

    ):

        self.Q=Q

        self.R=R

    def evaluate(

            self,

            x,

            u

    ):

        x=np.asarray(x)

        u=np.asarray(u)

        return (

            x.T@self.Q@x

            +

            u.T@self.R@u

        )
class Costate:

    """
    λ(t)

    """

    def __init__(

            self,

            dimension

    ):

        self.value=np.zeros(

            dimension

        )

    def set(

            self,

            lam

    ):

        self.value=np.asarray(

            lam

        )

class Hamiltonian:

    """
    H=L+λf

    """

    def __init__(

            self,

            system,

            running_cost

    ):

        self.system=system

        self.running_cost=running_cost

    def evaluate(

            self,

            x,

            u,

            lam

    ):

        L=self.running_cost.evaluate(

            x,

            u

        )

        f=self.system.derivative(

            0,

            x,

            u

        )

        return L+lam@f

class HamiltonianGradient:

    def __init__(

            self,

            hamiltonian,

            eps=1e-6

    ):

        self.ham=hamiltonian

        self.eps=eps

    def gradient_u(

            self,

            x,

            u,

            lam

    ):

        g=np.zeros_like(u)

        h0=self.ham.evaluate(

            x,

            u,

            lam

        )

        for i in range(

                len(u)

        ):

            up=u.copy()

            up[i]+=self.eps

            hp=self.ham.evaluate(

                x,

                up,

                lam

            )

            g[i]=(hp-h0)/self.eps

        return g

class OptimalController:

    """
    arg min H

    """

    def __init__(

            self,

            gradient,

            lr=0.05,

            iterations=100

    ):

        self.gradient=gradient

        self.lr=lr

        self.iterations=iterations

    def optimize(

            self,

            x,

            u0,

            lam

    ):

        u=u0.copy()

        for _ in range(

                self.iterations

        ):

            g=self.gradient.gradient_u(

                x,

                u,

                lam

            )

            u-=self.lr*g

        return u

Q=np.eye(

    len(x.values())

)

R=np.eye(

    len(u)

)

cost=RunningCost(

    Q,

    R

)

ham=Hamiltonian(

    graph_system.system,

    cost

)

lam=np.ones(

    len(x.values())

)

grad=HamiltonianGradient(

    ham

)

controller=OptimalController(

    grad

)

u_star=controller.optimize(

    x.values(),

    np.zeros(

        len(u)

    ),

    lam

)

print()

print(

    "Optimal Control"

)

print(

    u_star

)

# ============================================================
# Phase 11
# Pontryagin Maximum Principle (Prototype)
# ============================================================

class PMPSystem:

    """
    Combined State + Costate Dynamics

        x_dot = f(x,u)

        lambda_dot = - dH/dx
    """

    def __init__(self,
                 system,
                 hamiltonian,
                 eps=1e-6):

        self.system=system
        self.ham=hamiltonian
        self.eps=eps

    # --------------------------------------------------

    def costate_derivative(
            self,
            x,
            u,
            lam):

        n=len(x)

        grad=np.zeros(n)

        h0=self.ham.evaluate(
            x,
            u,
            lam
        )

        for i in range(n):

            xp=x.copy()
            xp[i]+=self.eps

            hp=self.ham.evaluate(
                xp,
                u,
                lam
            )

            grad[i]=(hp-h0)/self.eps

        return -grad

    # --------------------------------------------------

    def derivative(
            self,
            t,
            z,
            u):

        n=len(z)//2

        x=z[:n]

        lam=z[n:]

        dx=self.system.derivative(
            t,
            x,
            u
        )

        dlam=self.costate_derivative(
            x,
            u,
            lam
        )

        return np.concatenate(
            [
                dx,
                dlam
            ]
        )

class PMPSolver:

    def __init__(
            self,
            pmp):

        self.pmp=pmp

    def solve(
            self,
            x0,
            lambda0,
            u,
            tf=10):

        z0=np.concatenate(

            [

                x0,

                lambda0

            ]

        )
