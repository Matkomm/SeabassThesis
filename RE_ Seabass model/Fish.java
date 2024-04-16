import java.util.ArrayList;
import java.util.Random;

public class Fish {
/*Variables*/
	//state variables
	private float x;
	private float[] r;
	private float [] rDotPrev;
	private int behaviouralMode;
	
	//external objects
	private Environment environment;
	private Random randomGenerator;
	private ArtiFish2_0Main mainThread;
	
	//auxiliary variables
	int pelletsEaten = 0;
	private int satiationWaitingTimer = 0;
	private int manipulationTimer = 0;
	private int index;
	private float prevTemp = -1;
	private float prevZPos = 0;

/*Parameters*/	
	//physiological parameters
	private float bodyLength;
	private float bodyWeight;
	private float stomachVolume;
	private float maxSatiationWaitTime = 60;
	private float maxManipulateTime = Math.round(5.5f+Math.random()*(30.1f-5.5f));
	
	
	//behavioural parameters
	private float characteristicVelocity;
	//TODO: following four parameters determine the temperature response. Change for cod.
	private float tempHigh = 18;
	private float tempLow = 16;
	private float tempUILT = 82.0205f;
	private float tempLILT = -19.8022f;
	//TODO: following four parameters determine the light response. Change for cod.
	private float lightThresholdLow = (1.1f+(float)Math.random()*(3.4f));
	private float lightThresholdHigh = lightThresholdLow+2.5f;
	private float lightThresholdLL = lightThresholdLow-lightThresholdLow/0.3921f;
	private float lightThresholdHH = (2050.6f-lightThresholdHigh)/5.0781f+lightThresholdHigh;
	private float avoidanceBoundary;
	private float parallelBoundary;
	private float maxDirectionChange = 1f/3f;
	
	//other parameters
	private double pi;		
	
	
	public Fish(float BW, float BL,Environment env,float[] r0,int ind,ArtiFish2_0Main main){
		pi = Math.PI;
		mainThread = main;
		bodyLength = BL;
		bodyWeight = BW;
		//TODO: max stomach volume. Change for cod.
		stomachVolume = 0.0007f*(float)Math.pow(bodyWeight,1.3796);
		environment = env;
		r = new float[5];
		rDotPrev = new float[3];
		behaviouralMode = 0;
		x = stomachVolume/2;
		r[0] = r0[0];
		r[1] = r0[1];
		r[2] = r0[2];
		r[3] = r0[3];
		r[4] = r0[4];
		randomGenerator = new Random();
		//TODO: characteristic velocity. Change for cod. HTI
		characteristicVelocity = 0.5481f*bodyLength+0.0591f*bodyLength*(float)randomGenerator.nextGaussian();
		index = ind;
		
		//TODO: following two parameters determine the social behaviour. Change for cod.
		avoidanceBoundary = 0.66f*bodyLength;
		parallelBoundary = 3*bodyLength;	

	}
	public float[] fcn(int t, int[] binLatticeLocation,int decimation){
		//initialising auxiliary variables
		float[] specificOutput;
		float [] rDot = new float[3];
		float[] derivative = new float[5];

		//reading local environmental data
		float [] cageData = environment.returnCage(r);
		float[] temperatureData = environment.returnTemperature(r[2], t);
		float temperature = temperatureData[0];
		float localLightIntensity = environment.returnLight(r[2], t);
		
		//initialising quotas
		float horizontalQuota = 1;
		float verticalQuota = 1;

		//cage behaviour
		specificOutput = cageBehaviour(cageData);
		if(specificOutput[0]+specificOutput[1]!=0){
			rDot[0] += Math.min(horizontalQuota, specificOutput[3])*specificOutput[0]/(Math.abs(specificOutput[0]+specificOutput[1]));
			rDot[1] +=Math.min(horizontalQuota, specificOutput[3])*specificOutput[1]/(Math.abs(specificOutput[0]+specificOutput[1]));;
		}
		rDot[2] +=Math.min(verticalQuota, specificOutput[4])*Math.signum(specificOutput[2]);

		//reducing quotas
		horizontalQuota -= specificOutput[3];
		verticalQuota -= specificOutput[4];
		if(horizontalQuota<= 0){
			horizontalQuota = 0;
		}
		if(verticalQuota<= 0){
			verticalQuota = 0;
		}
//		food behaviour
		float [] foodData = environment.returnFood(r, t);
		float swSpeedIncF;

		if((behaviouralMode>=2)&&(behaviouralMode<5)){
			swSpeedIncF = 1.4f;
		}
		else{
			swSpeedIncF = 1;
		}
		specificOutput = foodBehaviour(foodData,temperature,r,t);
		rDot[0] += specificOutput[0]*Math.min(horizontalQuota, specificOutput[3]);//*swSpeedIncF;
		rDot[1] += specificOutput[1]*Math.min(horizontalQuota, specificOutput[3]);//*swSpeedIncF;
		rDot[2] += specificOutput[2]*Math.min(verticalQuota, specificOutput[4]);//*swSpeedIncF;

//		reducing quotas
		horizontalQuota -= specificOutput[3];
		verticalQuota -= specificOutput[4];
		if(horizontalQuota<= 0){
			horizontalQuota = 0;
		}
		if(verticalQuota<= 0){
			verticalQuota = 0;
		}
		if(behaviouralMode<=1){
			//temperature behaviour
			specificOutput = temperatureBehaviour(temperatureData);
			rDot[2] += Math.min(verticalQuota, specificOutput[3])*Math.signum(specificOutput[2]);
			//reducing quotas
			verticalQuota -= specificOutput[3];
			if(verticalQuota<= 0){
				verticalQuota = 0;
			}

			//light behaviour
			specificOutput = lightBehaviour(localLightIntensity);
			rDot[2] += Math.min(verticalQuota, specificOutput[3])*Math.signum(specificOutput[2]);

			//reducing quotas
			verticalQuota -= specificOutput[3];

			if(verticalQuota<= 0){
				verticalQuota = 0;
			}
		}
		//neighbour behaviour
		specificOutput = neighbourBehaviour(binLatticeLocation,cageData);
		if(specificOutput[0]+specificOutput[1]!=0){
			rDot[0] += Math.min(horizontalQuota, specificOutput[3])*specificOutput[0]/(Math.abs(specificOutput[0]+specificOutput[1]));
			rDot[1] +=Math.min(horizontalQuota, specificOutput[3])*specificOutput[1]/(Math.abs(specificOutput[0]+specificOutput[1]));;
		}
		rDot[2] +=Math.min(verticalQuota, specificOutput[4])*Math.signum(specificOutput[2]);

		//reducing quotas
		horizontalQuota -= specificOutput[3];
		verticalQuota -= specificOutput[4];
		if(horizontalQuota<= 0){
			horizontalQuota = 0;
		}
		if(verticalQuota<= 0){
			verticalQuota = 0;
		}

		//stochastic behaviour
		float maxval = 1.0f;
		specificOutput = stochasticBehaviour();

		rDot[0] += specificOutput[0]*Math.min(horizontalQuota,maxval);
		rDot[1] += specificOutput[1]*Math.min(horizontalQuota,maxval);
		rDot[2] += specificOutput[2]*Math.min(verticalQuota,maxval);

		//keep velocity within reasonable range
		float [] tempSpeed = {rDot[0],rDot[1],rDot[2]};
		if(ToolBox.normFloatArray(tempSpeed)>1){
			rDot[0]/=ToolBox.normFloatArray(tempSpeed);
			rDot[1]/=ToolBox.normFloatArray(tempSpeed);
			rDot[2]/=ToolBox.normFloatArray(tempSpeed);
		}

		//assembling output array
		float k1 = 0.35f;
		float k2 = 1-k1;
		rDot[0] = k1*rDot[0]+k2*rDotPrev[0];
		rDot[1] = k1*rDot[1]+k2*rDotPrev[1];
		rDot[2] = k1*rDot[2]+k2*rDotPrev[2];
		
		//changing swimming velocities due to light and food
		float swVelThr = 4f;
		float swLightRedFactor;
		if(localLightIntensity<=swVelThr){
			swLightRedFactor = 1-0.5f*(swVelThr-localLightIntensity)/swVelThr;
		}
		else{
			swLightRedFactor = 1;
		}
		rDot[0] = swLightRedFactor*swSpeedIncF*rDot[0];
		rDot[1] = swLightRedFactor*swSpeedIncF*rDot[1];
		rDot[2] = swLightRedFactor*swSpeedIncF*rDot[2];
		
		rDotPrev = rDot;
		derivative = microBehaviour(rDot);

		//updating position of fish
		r[0] += derivative[0];
		r[1] += derivative[1];
		r[2] += derivative[2];
		r[3] += derivative[3];
		r[4] += derivative[4];
		
		//safeguard to make sure that the orientation angles are between -pi and pi (saves time)
		if(r[3]>pi){
			r[3]-=2*pi;
		}
		else if(r[3]<-pi){
			r[3]+=2*pi;
		}
		if(r[4]>pi){
			r[4]-=2*pi;
		}
		else if(r[4]<-pi){
			r[4]+=2*pi;
		}

		//writing to velocity output matrix
		if(t%decimation==0){
			mainThread.rDot1Out[index][t/decimation] = derivative[0];
			mainThread.rDot2Out[index][t/decimation] = derivative[1];
			mainThread.rDot3Out[index][t/decimation] = derivative[2];
		}
		float [] output = {r[0],r[1],r[2],r[3],r[4],x};
//		if((index == 100)&&(t == 999)){
//			ArrayList<Fish> diddeliFish =  findNeighbourList(binLatticeLocation,cageData);
//			for (int i = 0;i<diddeliFish.size();i++){
//				System.out.println("Fish "+index+" has this neighbour "+diddeliFish.get(i).getIndex());
//			}
//		}
		return output;
	}
	private float[] cageBehaviour(float[] cageData){
		//reading out distances to cage and centre of cage
		float distanceToSurface = cageData[0];
		float distanceToWall = cageData[1];
		float distanceToBottom = cageData[2];
		float prefDistToSurf = 0.25f;
		float prefDistToWall = 1;
		float prefDistToBot = 0.25f;


		//creating proportion factors and output vectors
		float pSurf = 0;
		float pWall = 0;
		float pBottom = 0;
		float [] outputVector = new float[3];
		float [] outputQuotas = new float[2];

		//surface response
		if(distanceToSurface<=0){
			outputVector[2] = -1;
			pSurf = 1;
		}
		else if(distanceToSurface <=	prefDistToSurf){
			outputVector[2] += -(prefDistToSurf-distanceToSurface);
			pSurf = (prefDistToSurf-distanceToSurface);
		}

		//cage bottom response
		if(distanceToBottom <=0){
			outputVector[2] =1;
			pBottom = 1;
		}
		else if(distanceToBottom<=prefDistToBot){
			outputVector[2] += (prefDistToBot-distanceToBottom);
			pBottom = prefDistToBot-distanceToBottom;
		}

		//cage wall response
		if(distanceToWall<=0){
			float [] cageCentre = {cageData[3],cageData[4]};
			float [] vectorToCentre = {cageCentre[0]-r[0],cageCentre[1]-r[1]};
			for (int i = 0;i<vectorToCentre.length;i++){
				outputVector[i] += vectorToCentre[i]/ToolBox.normFloatArray(vectorToCentre);
			}
			pWall = 1.0f;
		}
		else if(distanceToWall <=prefDistToWall){
			float [] cageCentre = {cageData[3],cageData[4]};
			float [] vectorToCentre = {cageCentre[0]-r[0],cageCentre[1]-r[1]};
			for (int i = 0;i<vectorToCentre.length;i++){
				outputVector[i] += (prefDistToWall-distanceToWall)*vectorToCentre[i]/ToolBox.normFloatArray(vectorToCentre);
			}
			pWall = 1.0f-prefDistToWall;
		}


		outputQuotas[0] = 1.0f*pWall;
		outputQuotas[1] = 1.0f*Math.max(pSurf, pBottom);
		float[] output = {outputVector[0],outputVector[1],outputVector[2],outputQuotas[0],outputQuotas[1]};
		return output;
	}
	private float[] foodBehaviour(float [] foodData,float temperature,float[] r, int t){
		/**
		 * foodBehaviour: function to compute the response toward the presence of food in
		 * the sea-cage.
		 * Outputs: outputVector[0-2] = movement vector in relation to food
		 * 			outputVector[3] = proportion taken by horizontal quota
		 * 			outputVector[4] = proportion taken by vertical quota
		 * */
		//reading out values from input and creating arrays for output
		float [] outputVector = new float[5];
		float [] vectorToFood = new float[3];
		vectorToFood[0] = foodData[0];
		vectorToFood[1] = foodData[1];
		vectorToFood[2] = foodData[2];
		float rho = foodData[3];
		float pelletVolume = foodData[4];
		float feedingRadius = foodData[5];
		float changeInStomachContent = 0;
		float distanceToFood = 0;
		
		float[] vectorToFoodVerticalAxis = {vectorToFood[0],vectorToFood[1]};
		if(ToolBox.normFloatArray(vectorToFoodVerticalAxis)<=feedingRadius){
			distanceToFood = Math.abs(r[2]);
		}
		else{
			distanceToFood = (float)Math.sqrt(Math.pow(ToolBox.normFloatArray(vectorToFoodVerticalAxis)-feedingRadius,2)+Math.pow(r[2], 2));
		}
	

		//deriving present behavioural mode
		behaviouralMode = newBehaviouralMode(rho,r,distanceToFood,ToolBox.normFloatArray(vectorToFoodVerticalAxis),feedingRadius,pelletVolume);

		if(behaviouralMode<=1){
			//normal(0) or satiated(1) mode: the fish has not discovered
			//food(0) or is not hungry(1). Behaviour toward other factors as
			//normal
		}
		else if (behaviouralMode == 2){
			//approach mode: the fish has started to approach the
			//feeding area. Behaviour toward other factors reduced
			//to 0 in vertical direction and a minimum horizontally

			if(distanceToFood>=1){
				outputVector[0] = vectorToFood[0]/distanceToFood;
				outputVector[1] = vectorToFood[1]/distanceToFood;
				outputVector[2] = vectorToFood[2]/distanceToFood;
			}
			else{
				outputVector[0] = vectorToFood[0];
				outputVector[1] = vectorToFood[1];
				outputVector[2] = vectorToFood[2];

			}
			outputVector[3] = 0.9f;
			outputVector[4] = 0.9f;
		}
		else if (behaviouralMode == 3){
			//select mode: the fish has selected a pellet for ingestion. 
			//Behaviour toward other factors reduced to 0 in vertical 
			//direction and a minimum horizontally
			outputVector[3] = 0.9f;
			outputVector[4] = 0.9f;
			changeInStomachContent += pelletVolume;
			pelletsEaten++;
		}
		else {
			//manipulate mode: the fish has seized a pellet and starts
			//manipulating it. Behaviour toward other factors reduced to
			//0 in vertical direction but normal in horizontal direction

			outputVector[4] = 0.9f;
		}
		if(x > 0){
			float a = (float)(5.2591*Math.pow(10f,-6f));
			float b = (float)Math.pow(temperature,0.7639f);
			changeInStomachContent -= a*x*b;
		}
		x = x+changeInStomachContent;
		return outputVector;		
	}
	
	private float[] temperatureBehaviour(float[] temperatureData){
		float [] outputVector = new float[4];
		float temperature = temperatureData[0];
		float tempDirectivity;

		//new approach; using thermal history
		if(prevTemp != -1){
			if(temperature>prevTemp){
				tempDirectivity = Math.signum(r[2]-prevZPos);
			}
			else if(temperature<prevTemp){
				tempDirectivity = Math.signum(prevZPos-r[2]);
			}
			else{
				tempDirectivity = 0;
			}
		}
		else{
			tempDirectivity = 0;
		}

		if(tempDirectivity != 0){
			if(temperature > tempHigh){
				outputVector[3] = (temperature-tempHigh)/(tempUILT-tempHigh);
				outputVector[2] = -tempDirectivity*outputVector[3];
			}
			else if(temperature<tempLow){
				outputVector[3] = (tempLow-temperature)/(tempLow-tempLILT);
				outputVector[2] = tempDirectivity*outputVector[3];
			}
		}
		prevTemp = temperature;
		prevZPos = r[2];
		return outputVector;
	}

	private float[] lightBehaviour(float localLightIntensity){
		float [] outputVector = new float[4];
		float relativeLightIntensity = localLightIntensity;


		if(relativeLightIntensity>lightThresholdHigh){
			outputVector[2] = -(relativeLightIntensity-lightThresholdHigh)/(lightThresholdHH-lightThresholdHigh);
			outputVector[3] = -outputVector[2];
		}
		else if(relativeLightIntensity<lightThresholdLow){
			
			outputVector[2] = (lightThresholdLow-relativeLightIntensity)/(lightThresholdLow-lightThresholdLL);
			outputVector[3] = outputVector[2];
		}
		else{
			//do nothing
		}

		return outputVector;
	}

	private float[] neighbourBehaviour(int[] binLatticeLocation,float[] cageData){
		ArrayList<Fish> neighbourList;
		float [] outputVector = new float[5];
		neighbourList = findNeighbourList(binLatticeLocation,cageData);
		int n = neighbourList.size();
		int m = 0;
		
		for (int i = 0;i<n;i++){
			float [] tempOut = new float[3];
			int neighbourIndex = neighbourList.get(i).getIndex();
			if(neighbourIndex!=index){
				float[] vectorToNeighbour = {neighbourList.get(i).getPosition()[0]-r[0],neighbourList.get(i).getPosition()[1]-r[1],neighbourList.get(i).getPosition()[2]-r[2]};
				float distanceToNeighbour = ToolBox.normFloatArray(vectorToNeighbour);
				if(distanceToNeighbour<avoidanceBoundary ){
					if(distanceToNeighbour>0){
						tempOut[0] = -vectorToNeighbour[0]*(avoidanceBoundary -distanceToNeighbour)/avoidanceBoundary;
						tempOut[1] = -vectorToNeighbour[1]*(avoidanceBoundary -distanceToNeighbour)/avoidanceBoundary;
						tempOut[2] = -vectorToNeighbour[2]*(avoidanceBoundary -distanceToNeighbour)/avoidanceBoundary;
					}
					else{
						tempOut[0] = (float)randomGenerator.nextGaussian()*0.05f;
						tempOut[1] = (float)randomGenerator.nextGaussian()*0.05f;
						tempOut[2] = (float)randomGenerator.nextGaussian()*0.05f;
					}
				}
				else if(distanceToNeighbour<parallelBoundary){
					float []rDotPrevTemp = neighbourList.get(i).getVelocity();
					float velocityMagnitude = ToolBox.normFloatArray(rDotPrevTemp);
					if(velocityMagnitude>0){
						rDotPrevTemp[0] =rDotPrevTemp[0]/velocityMagnitude; 
						rDotPrevTemp[1] =rDotPrevTemp[1]/velocityMagnitude;
						rDotPrevTemp[2] =rDotPrevTemp[2]/velocityMagnitude;
					}
					float factor = 0.35f;
					tempOut[0] = rDotPrevTemp[0]*factor*(parallelBoundary-distanceToNeighbour)/(parallelBoundary-avoidanceBoundary);
					tempOut[1] = rDotPrevTemp[1]*factor*(parallelBoundary-distanceToNeighbour)/(parallelBoundary-avoidanceBoundary);
					tempOut[2] = rDotPrevTemp[2]*factor*(parallelBoundary-distanceToNeighbour)/(parallelBoundary-avoidanceBoundary);
				}
			}
			if(ToolBox.normFloatArray(tempOut)>0){
				outputVector[0] = outputVector[0]+tempOut[0];
				outputVector[1] = outputVector[1]+tempOut[1];
				outputVector[2] = outputVector[2]+tempOut[2];
				m +=1;
			}
		}
		if(m>0){
			outputVector[0] = outputVector[0]/m;
			outputVector[1] = outputVector[1]/m;
			outputVector[2] = outputVector[2]/m;
		}
		float [] temp = {outputVector[0],outputVector[1]};
		outputVector[3] = ToolBox.normFloatArray(temp);
		outputVector[4] = Math.abs(outputVector[2]);
		return outputVector;
	}
		private float[] stochasticBehaviour(){
		float psi = r[3];
		float theta = r[4];

		float cosPsi = (float)Math.cos(psi);
		float sinPsi = (float)Math.sin(psi);
		float cosTheta = (float)Math.cos(theta);
		float sinTheta = (float)Math.sin(theta);

		float[][] R = new float[3][3];
		R[0][0] = cosPsi*cosTheta;
		R[0][1] = -sinPsi;
		R[0][2] = -cosPsi*sinTheta;
		R[1][0] = sinPsi*cosTheta;
		R[1][1] = cosPsi;
		R[1][2] = sinPsi*sinTheta;
		R[2][0] = sinTheta;
		R[2][1] = 0;
		R[2][2] = cosTheta;

		float directionVariance = 0.25f;
		float [] outDirection = {1.0f,directionVariance*(float)randomGenerator.nextGaussian(),directionVariance*(float)randomGenerator.nextGaussian()};
		float outDirectionNorm = ToolBox.normFloatArray(outDirection);
		float outAmplitude = 1;
		for (int i = 0;i<3;i++){
			outDirection[i] = outAmplitude*outDirection[i]/outDirectionNorm;  
		}
		float [] outputVector = ToolBox.matrixTimesVector(R, outDirection);

		return outputVector;
	}
	private float[] microBehaviour(float[] rDot){
		float psi = r[3];
		float theta = r[4];
		float psiRef = (float)Math.atan2(rDot[1], rDot[0]);
		float thetaRef = (float)Math.atan2(rDot[2], rDot[0]/Math.cos(psiRef));

		//deriving values of psi and theta within the appropriate range (-pi to pi)
		if(psi>0){
			while (psi>pi){
				psi -=2*pi;
			}
		}
		else {
			while (psi<-pi){
				psi +=2*pi;
			}
		}
		if(theta>0){
			while (theta>pi){
				theta -=2*pi;
			}
		}
		else {
			while (theta<-pi){
				theta +=2*pi;
			}
		}

		//creating angular derivatives; making sure that they are within the appropriate
		//range
		float psiDot = psiRef-psi;
		float thetaDot = thetaRef-theta;

		if(psiDot>0) {
			while (psiDot>pi){
				psiDot -=2*pi;
			}
		}
		else {
			while (psiDot<-pi){
				psiDot +=2*pi;
			}
		}
		if (thetaDot>0){
			while (thetaDot>pi){
				thetaDot -=2*pi;
			}
		}
		else  {
			while (thetaDot<-pi){
				thetaDot +=2*pi;
			}
		}

		//making sure that fish turns in the appropriate direction
		double s = 0;
		if(psiDot<0){
			if(psiDot<-pi){
				s = -1;
			}
			else {
				s = 1;
			}
		}
		else {
			if(psiDot>pi){
				s = -1;
			}
			else {
				s = 1;
			}
		}
		psiDot = psiDot*(float)s*maxDirectionChange;

		if(thetaDot<0){
			if(thetaDot<-pi){
				s = -1;
			}
			else {
				s = 1;
			}
		}
		else {
			if(thetaDot>pi){
				s = -1;
			}
			else {
				s = 1;
			}
		}
		thetaDot = thetaDot*(float)s*maxDirectionChange;

		//constructing rotation matrtix from system locat to fish to inertial frame
		psi = psi+psiDot;
		theta = theta+thetaDot;
		float cosPsi = (float)Math.cos(psi);
		float sinPsi = (float)Math.sin(psi);
		float cosTheta = (float)Math.cos(theta);
		float sinTheta = (float)Math.sin(theta);
		float [][] R = {{cosPsi*cosTheta,-sinPsi,-cosPsi*sinTheta},{sinPsi*cosTheta,cosPsi,sinPsi*sinTheta},{sinTheta,0,cosTheta}};

		//impose constraints on fish movement
		float[] rTemp = ToolBox.matrixTimesVector(ToolBox.transposeFloatMatrix(R), rDot);
		rTemp[0] = rTemp[0]*characteristicVelocity/0.8f;
		rTemp[1] = 0;
		rTemp[2] = 0;
		rTemp = ToolBox.matrixTimesVector(R, rTemp);
		float[] output = {rTemp[0],rTemp[1],rTemp[2],psiDot,thetaDot};
		return output;
	}
	private ArrayList<Fish> findNeighbourList(int[] binLatticeLocation, float[] cageData){
		ArrayList<Fish> neighbourList = new ArrayList<Fish>();
		float [] cageCentre = {cageData[0],cageData[1]};
		float cageRadius = cageData[2];
		int binLatticeX = binLatticeLocation[0];
		int binLatticeY = binLatticeLocation[1];
		int binLatticeZ = binLatticeLocation[2];
		float cellWidth = mainThread.cellWidth;
		float delta = mainThread.delta;
		float x = r[0];
		float y = r[1];
		float z = r[2];
		int hRange = mainThread.horRange;
		int vRange = mainThread.verRange;

		//creating side panels
		float sideWest = cageCentre[0]-cageRadius-delta+cellWidth*(binLatticeX-1);
		float sideEast = sideWest+cellWidth;
		float sideSouth = cageCentre[1]-cageRadius-delta+cellWidth*(binLatticeY-1);
		float sideNorth = sideSouth+cellWidth;
		float sideUp = cellWidth*(1-binLatticeZ);
		float sideDown = cellWidth*(-binLatticeZ);

		//creating corners
		float [][] corners = {{sideWest,sideSouth,sideUp},{sideWest,sideNorth,sideUp},{sideEast,sideSouth,sideUp},{sideEast,sideNorth,sideUp},{sideWest,sideSouth,sideDown},{sideWest,sideNorth,sideDown},{sideEast,sideSouth,sideDown},{sideEast,sideNorth,sideDown}};

		//creating edge lines
		float [][] edges = {{x, sideSouth,sideUp},{sideWest, y, sideUp},{sideWest, sideSouth, z},{x, sideNorth, sideUp},{sideEast,y,sideUp},{sideEast,sideNorth,z},{x, sideSouth, sideDown},{sideEast, y ,sideDown},{sideEast,sideSouth,z},{x, sideNorth,sideDown},{sideWest,y,sideDown},{sideWest,sideNorth,z}};

		//adding fish in resident cell to neighbourList
		neighbourList.addAll(mainThread.binLattice[binLatticeX][binLatticeY][binLatticeZ]);

		//checking side panel criterions
		if(x-sideWest<=5*bodyLength){
			if(binLatticeX-1>=0){
				neighbourList.addAll(mainThread.binLattice[binLatticeX-1][binLatticeY][binLatticeZ]);
			}
		}
		if(sideEast-x<=5*bodyLength){
			if(binLatticeX+1<hRange){
				neighbourList.addAll(mainThread.binLattice[binLatticeX+1][binLatticeY][binLatticeZ]);
			}
		}
		if(y-sideSouth<=5*bodyLength){
			if(binLatticeY-1>=0){
				neighbourList.addAll(mainThread.binLattice[binLatticeX][binLatticeY-1][binLatticeZ]);
			}
		}
		if(sideNorth-y<= 5*bodyLength){
			if(binLatticeY+1<hRange){
				neighbourList.addAll(mainThread.binLattice[binLatticeX][binLatticeY+1][binLatticeZ]);
			}
		}
		if(sideUp-z<=5*bodyLength){
			if(binLatticeZ-1>=0){
				neighbourList.addAll(mainThread.binLattice[binLatticeX][binLatticeY][binLatticeZ-1]);
			}
		}
		if(z-sideDown<=5*bodyLength){
			if(binLatticeZ+1<vRange){
				neighbourList.addAll(mainThread.binLattice[binLatticeX][binLatticeY][binLatticeZ+1]);
			}
		}

		//checking corner criterions
		for (int i = 0;i<corners.length;i++){
			float[] distanceVector = {corners[i][0]-x,corners[i][1]-y,corners[i][2]-z};
			float distance = ToolBox.normFloatArray(distanceVector);
			if(distance<=5*bodyLength){
				int [] newCoord = {binLatticeX+(int)Math.signum(distanceVector[0]),binLatticeY+(int)Math.signum(distanceVector[1]),binLatticeZ+(int)Math.signum(-distanceVector[2])};
				if(((newCoord[0]>=0)&&(newCoord[0]<hRange))&&((newCoord[1]>=0)&&(newCoord[1]<hRange))&&((newCoord[2]>=0)&&(newCoord[2]<vRange))){
					neighbourList.addAll(mainThread.binLattice[newCoord[0]][newCoord[1]][newCoord[2]]);
				}
			}
		}
		//checking edge criterions
		for (int i = 0;i<edges.length;i++){
			float[] distanceVector = {edges[i][0]-x,edges[i][1]-y,edges[i][2]-z};
			float distance = ToolBox.normFloatArray(distanceVector);
			if(distance<=5*bodyLength){
				int [] newCoord = {binLatticeX+(int)Math.signum(distanceVector[0]),binLatticeY+(int)Math.signum(distanceVector[1]),binLatticeZ+(int)Math.signum(-distanceVector[2])};
				if(((newCoord[0]>=0)&&(newCoord[0]<hRange))&&((newCoord[1]>=0)&&(newCoord[1]<hRange))&&((newCoord[2]>=0)&&(newCoord[2]<vRange))){
					neighbourList.addAll(mainThread.binLattice[newCoord[0]][newCoord[1]][newCoord[2]]);
				}
			}
		}

		return neighbourList;
	}
	public float getBodyLength(){
		return bodyLength;
	}
	public float getBodyWeight(){
		return bodyWeight;
	}
	public float getStomachVolume(){
		return stomachVolume;
	}
	public int getPelletsEaten(){
		return pelletsEaten;
	}
	public float[] getPosition(){
		return r;
	}
	public float[] getVelocity(){
		return rDotPrev;
	}
	public int getIndex(){
		return index;
	}
	public int getBehaviouralMode(){
		return behaviouralMode;
	}
	private int newBehaviouralMode(float rho,float[] r,float distanceToFood,float horizontalDistanceToFood, float feedingRadius,float pelletVolume){

		//derive values from input and initialise output
		int newMode = 0;
		float xNorm = x/stomachVolume;

		
		//define different probabilities
		float detectionProbability;
		float captureProbability;
		float hungerProbability;

		detectionProbability =1/(1+distanceToFood);
		float a = 0.05f;
		float b = 3;
		if(horizontalDistanceToFood<=feedingRadius){
			captureProbability = a*1/(float)Math.pow(1+distanceToFood,b);
		}else{
			captureProbability = 0;
		}
		
		if(xNorm>0.3){
			hungerProbability = 0.5f-0.57f*(xNorm-0.3f)/(xNorm-0.2f);
		}
		else {
			if(xNorm >=1){
				hungerProbability = -1.0f;
			}
			else {
				hungerProbability = 0.5f+0.67f*(0.3f-xNorm)/(0.4f-xNorm);
			}
		}

		if(captureProbability<0){
			captureProbability = 0;
		}
		if(hungerProbability<0){
			hungerProbability = 0;
		}


		if(pelletVolume == 0){
			newMode = 0;
		}
		else if(x+pelletVolume>stomachVolume){
			newMode = 1;
		}
		else {
			if(behaviouralMode == 0){
				//normal mode: no knowledge of available food
				if(Math.random()<detectionProbability){
					// food detected by fish
					if(Math.random()<hungerProbability){
						// hungry
						newMode = 2;
					}
					else {
						// satiated
						newMode = 1;
					}
				}
				else {
					// food not detected by fish
					newMode = 0;
				}
			}
			else if (behaviouralMode == 1){
				// satiated mode: food detected but fish not hungry
				if(satiationWaitingTimer>=maxSatiationWaitTime){
					if(Math.random()<hungerProbability){
						// hungry
						newMode = 2;
						satiationWaitingTimer = 0;
					}
					else {
						satiationWaitingTimer = 0;
						newMode = 1;
					}
				}
				else {
					// satiated
					newMode = 1;
					satiationWaitingTimer++;
				}
			}
			else if (behaviouralMode == 2){
				// approach mode: food detected and fish hungry
				if(Math.random()<captureProbability){
					// successful capture of single pellet
					newMode = 3;
				}
				else {
					// unsuccessful capture of pellet;continue searching
					newMode = 2;
				}
			}
			else if (behaviouralMode == 3){
				// capture mode: fish has captured pellet and gets ready to ingest; 
				// direct transition to manipulate mode
				newMode = 4;
			}
			else {
				// manipulate mode: the fish stays in this mode for 12 sec after having
				// successfully acquired a pellet; then goes to either mode 2 or 1 depending
				// on hunger
				if (manipulationTimer<maxManipulateTime){
					// still manipulating pellet
					newMode = 4;
					manipulationTimer++;
				}
				else {
					// finished manipulating pellet
					manipulationTimer = 0;
					if(Math.random()<hungerProbability){
						// still hungry; will pursue another pellet
						newMode = 2;
					}
					else{
						// satiated; no interest in presently pursuing more pellets
						newMode = 1;
					}
				}
			}
		}
		return newMode;
	}
}

