class pose:
	x = None
	y = None
	z = None
	rx= None
	ry= None
	rz= None

	def __init__(self,pos=None):
		if pos == None: # if no initial pose is given, default is zero
			self.x = 0
			self.y = 0
			self.z = 0
			self.rx= 0
			self.ry= 0
			self.rz= 0
		else:
			self.x = pos[0]
			self.y = pos[1]
			self.z = pos[2]
			self.rx= pos[3]
			self.ry= pos[4]
			self.rz= pos[5]

	def __str__(self): # function for printing the pose class
		return 'Pose:\n\tx:  %i \n\ty:  %i\n\tz:  %i\n\trx: %.3f\n\try: %.3f\n\trz: %.3f'%(self.x,
				self.y, self.z, self.rx, self.ry, self.rz)

