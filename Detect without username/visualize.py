import turtle

turtle.setup(1200, 800)

#GENERATE A GRID
grid = turtle.Turtle()
grid.penup()
grid.speed(0)
grid.color("gray")
for i in range(0, 10) :
	grid.setpos(-500, -300)
	grid.penup()
	grid.left(90)
	grid.forward(i*60)
	grid.right(90)
	grid.pendown()
	grid.forward(1000)
	grid.penup()

colors  = ["red","green","blue","orange","purple","pink","yellow"]
colori = 0


#Read dataStore.txt
lines = [line.rstrip('\n') for line in open('dataStore.txt')]
for line in lines:
	splitLine = line.split()

	drawerTurtle = turtle.Turtle()

	if(colori > 6):
		colori = 0

	drawerTurtle.color(colors[colori])
	colori = colori + 1
	drawerTurtle.penup()
	drawerTurtle.setpos(-500, -300)
	drawerTurtle.pendown()

	prev = 0

	for i in range(1, len(splitLine)):
		#print("PREV  : {} CURR : {}".format(prev, splitLine[i])),
		if(float(splitLine[i]) > prev):
			#print("MOVEMENT UP : {}".format((float(splitLine[i]) - prev)))
			drawerTurtle.left(90)
			drawerTurtle.forward((float(splitLine[i]) - prev) * 50)
			drawerTurtle.right(90)
			drawerTurtle.forward(40)
		else :

			#print("MOVEMENT DOWN : {}".format((float(splitLine[i]) + prev)))
			drawerTurtle.right(90)
			drawerTurtle.forward((prev - float(splitLine[i])) * 50)
			drawerTurtle.left(90)
			drawerTurtle.forward(40)

		prev = float(splitLine[i])



turtle.getscreen()._root.mainloop()