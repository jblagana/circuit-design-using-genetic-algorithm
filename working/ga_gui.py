import PySimpleGUI as sg
import re
import ga as ourLib

import os
from pathlib import Path
from os import path

sg.theme('DarkBlue14')

layout= [   [sg.Text('Hey it\'s me')],
            [sg.Text('.asc file path'), sg.FileBrowse()],
            [sg.Text('Resistor name(s): '), sg.InputText()],
            [sg.Text('Capacitor name(s): '), sg.InputText()],
            [sg.Text('Inductor name(s): '), sg.InputText()],
            [sg.Text('Output name(s): '), sg.InputText()],
            [sg.Text('Desired output value: '), sg.InputText()],
            [sg.Text('Number of generations: '), sg.InputText()],
            [sg.Text('Number of solutions per generations: '), sg.InputText()],
            [sg.Button('Simulate'), sg.Button('Exit')]
]
pattern = re.compile(r"^(\w+)(,\s*\w+)*$")
ascKey="Browse"
resistKey=0
capKey = 1
inductorKey = 2
outputKey=3
valueKey=4
genKey=5
solNumKey=6

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

window=sg.Window('Circuit and GA Parameters',layout)

def parseInput(names):
    return names.replace(" ", "").split(",")

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    resistorInput=values[resistKey]
    capacitorInput = values[capKey]
    inductorInput = values[inductorKey]
    ascPath=values[ascKey]
    output=values[outputKey]
    value=values[valueKey]
    gen=values[genKey]
    solNum=values[solNumKey]

    if not pattern.match(resistorInput) and not resistorInput=="":
        sg.popup("Invalid resistors")
        continue
    elif resistorInput == "NA":
        resistors = []
    elif not resistorInput=="":
        resistors=parseInput(resistorInput)
    
    if not pattern.match(capacitorInput) and not capacitorInput=="":
        sg.popup("Invalid capacitors")
        continue
    elif capacitorInput == "NA":
        capacitors = []
    elif not capacitorInput=="":
        capacitors=parseInput(capacitorInput)

    if not pattern.match(inductorInput) and not inductorInput=="":
        sg.popup("Invalid inductors")
        continue
    elif inductorInput == "NA":
        inductors = []
    elif not capacitorInput=="":
        inductors=parseInput(inductorInput)

    if not is_number(value):
        sg.popup("Invalid output value")
        continue

    if not is_number(gen):
        sg.popup("Invalid generation")
        continue

    if not is_number(solNum):
        sg.popup("Invalid number of solutions per generation")
        continue

    if not Path(ascPath).is_file() or not Path(ascPath).suffix==".asc":
        sg.popup("Invalid path")
    else:
        ourLib.ga_sim(ascPath,resistors,capacitors,inductors,output,value,gen,solNum)
