import java.sql.Time;
import java.util.*;

public class ArtiFish2_0Main extends Thread{

	/**
	 * @param args
	 */
	private Environment environment;
	private float meanBodyWeightOrLength;
	private float weightOrLengthStandardDeviation;
	private char fishSizeFormat;
	private int numberOfFish;
	private float [] cageVector;
	public ArrayList <Fish> fishList;
	private int timeHorizon;
	private int timeStep;
	public ArrayList<Fish>[][][] binLattice;
	private float maxBodyLength;
	public float [][] rOutTemp;
	private int [][] binLatticeLocations;
	public int horRange;
	public int verRange;
	public float delta;
	public float cellWidth;
	private double pi;
	public float [][] r1Out;
	public float [][] r2Out;
	public float [][] r3Out;
	public float [][] r4Out;
	public float [][] r5Out;
	public float [][] rDot1Out;
	public float [][] rDot2Out;
	public float [][] rDot3Out;
	public float [][] stomachContents;
	public float feedingFishProportion;
	public int numberOfFishFeeding;
	private String scenarioName;
	private int period;
	private char treatment;
	private int decimation;
	
	public ArtiFish2_0Main(int numFish,int timeHor, int timeSt,String scenName, int per,char treatm,int decim){
		decimation = decim;
		numberOfFish = numFish;
		timeHorizon = timeHor;
		timeStep = timeSt;
		scenarioName = scenName;
		period = per;
		treatment = treatm;
		fishList = new ArrayList<Fish>();
		environment = new Environment();
		pi = Math.PI;

		r1Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		r2Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		r3Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		r4Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		r5Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		rDot1Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		rDot2Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		rDot3Out = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		stomachContents = new float[numberOfFish][(int)Math.floor(timeHorizon/(timeStep*decimation))];
		binLatticeLocations  = new int[numberOfFish][3];
		System.out.println("Initialisation process commencing");
	}
	public void selectScenario(){
		if(scenarioName == "Johansson et al. 2006"){
			//initialising cage
			cageVector = new float[4];
			cageVector[0] = 1;
			cageVector[1] = 1;
			cageVector[2] = (float)Math.sqrt(144/Math.PI);
			cageVector[3] = 14;
			environment.initCage(cageVector);

			//initialising fish parameters
			fishSizeFormat = 'W';
			meanBodyWeightOrLength = 1280f; 			//mean body weight in g
			weightOrLengthStandardDeviation = 20f;		//standard deviation in g
			
			//initialising food
			int[] isF = new int[2];
			isF[0] = 32400;
//			isF[0] = 200;
			isF[1] = 50400;
			int[] dsF = new int[2];
			dsF[0] = 10800;
//			dsF[0] = 300;
			dsF[1] = 7200;
			environment.initFood(isF, dsF);
						
			//initialising seasonally dependent factors
			if(period == 1){
				float lat = 61;
				int d = 253;
				float kA = 0.2684f;
				float cloudEffect = 0.5019f;
				float bgRad = 2.6f;
				environment.initLight(lat, d, kA,cloudEffect,bgRad);

				if(treatment == 'C'){
					float dzT = 0.5f;
					float[] temps = {17.3919f,17.3919f,18.0544f,18.8473f,19.2883f,19.3778f,19.3573f,19.2966f,19.2359f,19.1697f,19.1167f,19.0715f,19.0260f,18.9866f,18.9416f,18.8927f,18.8432f,18.7875f,18.7222f,18.6460f,18.5813f,18.4924f,18.4152f,18.3171f,18.2444f,18.1738f,18.0984f,17.9782f,17.8876f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Parameters collected from "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'H'){
					float dzT = 0.5f;
					float[] temps = {17.6855f,17.6855f,18.2522f,18.9558f,19.2903f,19.3249f,19.2800f,19.2265f,19.1771f,19.1302f,19.0822f,19.0444f,19.0020f,18.9634f,18.9210f,18.8816f,18.8414f,18.7941f,18.7388f,18.6650f,18.5958f,18.5292f,18.4430f,18.4430f,18.4430f,18.4430f,18.4430f,18.4430f,18.4430f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Parameters collected from "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'N'){
					float dzT = 0.5f;
					float[] temps = {17.6704f,17.6704f,18.3344f,19.0302f,19.3269f,19.3528f,19.3034f,19.2485f,19.1936f,19.1412f,19.0940f,19.0480f,19.0061f,18.9615f,18.9232f,18.8768f,18.8154f,18.7619f,18.7113f,18.6559f,18.5880f,18.5257f,18.4690f,18.4690f,18.4690f,18.4690f,18.4690f,18.4690f,18.4690f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Parameters collected from "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				//fix; getting temperatures correct according to figure in article
				float dzT = 0.5f;
				float[] temps = {16.6641f,17.3919f,18.0544f,18.8473f,19.2883f,19.3778f,19.3573f,19.2966f,19.2359f,19.1697f,19.1167f,19.0715f,19.0260f,18.9866f,18.9416f,18.8927f,18.8432f,18.7875f,18.7222f,18.6460f,18.5813f,18.4924f,18.4152f,18.3171f,18.2444f,18.1738f,18.0984f,17.9782f,17.8876f};
				environment.initTemperature(dzT,temps);
			}
			else if (period == 2){
				float lat = 61;
				int d = 271;
				float kA = 0.1750f;
				float cloudEffect = 0.0647f;
				float bgRad = 2.6f;
				environment.initLight(lat, d, kA,cloudEffect,bgRad);

				if(treatment == 'C'){
					float dzT = 0.5f;
					float[] temps = {13.2777f,13.2777f,13.4746f,13.8302f,14.3017f,14.6960f,15.0870f,15.5020f,15.9038f,16.1228f,16.2393f,16.3186f,16.3094f,15.9500f,15.2069f,14.3403f,13.5601f,12.8774f,12.3337f,11.9020f,11.5647f,11.3393f,11.1491f,10.9813f,10.8514f,10.7179f,10.7179f,10.7179f,10.7179f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'H'){
					float dzT = 0.5f;
					float[] temps = {13.4418f,13.4418f,13.5887f,13.9509f,14.3752f,14.7699f,15.1116f,15.5393f,15.8195f,16.0149f,16.1865f,16.2643f,16.2093f,15.7297f,14.9663f,14.1477f,13.4440f,12.8545f,12.3641f,12.0225f,11.6813f,11.4435f,11.2680f,11.0385f,10.8911f,10.7306f,10.7306f,10.7306f,10.7306f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'N'){
					float dzT = 0.5f;
					float[] temps = {13.4286f,13.4286f,13.5852f,13.9775f,14.3824f,14.8238f,15.2213f,15.5864f,15.8252f,16.0536f,16.3340f,16.4387f,16.3580f,15.7554f,14.9881f,14.2196f,13.3815f,12.7480f,12.2566f,11.8877f,11.5966f,11.3503f,11.0982f,10.9243f,10.7554f,10.6193f,10.6193f,10.6193f,10.6193f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				//fix; getting temperatures correct according to figure in article
				float dzT = 0.5f;
				float[] temps = {13.0014f,13.2777f,13.4746f,13.8302f,14.3017f,14.6960f,15.0870f,15.5020f,15.9038f,16.1228f,16.2393f,16.3186f,16.3094f,15.9500f,15.2069f,14.3403f,13.5601f,12.8774f,12.3337f,11.9020f,11.5647f,11.3393f,11.1491f,10.9813f,10.8514f,10.7179f,10.5862f,10.4536f,10.3215f};					
				environment.initTemperature(dzT,temps);
			}
			else if (period == 3){
				float lat = 61;
				int d = 278;
				float kA = 0.1888f;
				float cloudEffect = 0.3010f;
				float bgRad = 3.7f;
				environment.initLight(lat, d, kA,cloudEffect,bgRad);

				if(treatment == 'C'){
					float dzT = 0.5f;
					float[] temps = {12.6984f,12.6984f,14.3391f,15.4207f,15.7002f,15.5926f,15.3379f,15.0500f,14.7522f,14.4806f,14.2587f,14.0869f,13.9486f,13.8644f,13.7986f,13.7424f,13.6913f,13.6458f,13.5976f,13.5378f,13.4900f,13.4421f,13.3960f,13.3488f,13.3052f,13.3052f,13.3052f,13.3052f,13.3052f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'H'){
					float dzT = 0.5f;
					float[] temps = {12.5262f,12.5262f,13.7133f,14.9139f,15.3078f,15.2728f,15.0610f,14.8053f,14.6003f,14.3867f,14.2130f,14.0839f,13.9901f,13.8967f,13.8332f,13.7644f,13.7140f,13.6634f,13.6227f,13.5762f,13.5364f,13.4857f,13.4331f,13.3629f,13.3080f,13.3080f,13.3080f,13.3080f,13.3080f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				else if(treatment == 'N'){
					float dzT = 0.5f;
					float[] temps = {12.5619f,12.5619f,13.8670f,15.1100f,15.4136f,15.3314f,15.1030f,14.8456f,14.6049f,14.3791f,14.1893f,14.0495f,13.9347f,13.8517f,13.7888f,13.7263f,13.6699f,13.6201f,13.5717f,13.5241f,13.4826f,13.4335f,13.3845f,13.3443f,13.2926f,13.2926f,13.2926f,13.2926f,13.2926f};					
					environment.initTemperature(dzT,temps);
					System.out.println("Collecting parameters for "+scenarioName+" period "+period+" with treatment "+treatment);
				}
				//fix; getting temperatures correct according to figure in article
				float dzT = 0.5f;
				float[] temps = {11.3372f,12.6984f,14.3391f,15.4207f,15.7002f,15.5926f,15.3379f,15.0500f,14.7522f,14.4806f,14.2587f,14.0869f,13.9486f,13.8644f,13.7986f,13.7424f,13.6913f,13.6458f,13.5976f,13.5378f,13.4900f,13.4421f,13.3960f,13.3488f,13.3052f,13.2597f,13.2152f,13.1702f,13.1255f};					
				environment.initTemperature(dzT,temps);
			}
			else{
				System.out.println("Wrong scenario selected in Johansson et al. 2006. Shutting down");
				System.exit(1);
			}
		}
		else if (scenarioName == "Schooling Scenario"){
			//initialising cage
			cageVector = new float[4];
			cageVector[0] = 1;
			cageVector[1] = 1;
			cageVector[2] = 10;
			cageVector[3] = 10;
			environment.initCage(cageVector);

			//initialising fish parameters
			fishSizeFormat = 'L';
			meanBodyWeightOrLength = 0.45f; 					//mean body length in m
			weightOrLengthStandardDeviation = 0.05f;			//standard deviation in m
			
			//initialising temperature (no gradient)
			float dzT = -1.0f;
			float[] temps = {17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f,17f};					
			environment.initTemperature(dzT,temps);
			
			//initialising light
			float lat = 61;
			int d = 278;
			float kA = 0.15f;
			float cloudEffect = 1;
			float bgRad = 0;
			environment.initLight(lat, d, kA,cloudEffect,bgRad);
			//initialising food
			int[] isF = new int[2];
			isF[0] = 32400;
			isF[1] = 50400;
			int[] dsF = new int[2];
			dsF[0] = 10800;
			dsF[1] = 7200;
			environment.initFood(isF, dsF);
		}
		
		//assigning pellet sizes
 
		float b0 = -19.8710f;							//Wing et al. 1998
		float b1 = 3.2123f;								//Wing et al. 1998
		
		if(fishSizeFormat == 'W'){
			environment.setPelletSize(meanBodyWeightOrLength);
		}
		
		else{
			environment.setPelletSize((float) (Math.exp(b0)*Math.pow(meanBodyWeightOrLength*1000,b1))*1000);
		}

	}
	public void initialiseModel(){
		//timeStep = 1;
		Random randomGenerator = new Random();
		selectScenario();
		rOutTemp =  new float[numberOfFish][3];

		//creating fish objects
		float BW = 0;
		float BL = 0;
		float[] r0 = new float[5];
		maxBodyLength = 0;
		double alpha;
		for (int i = 0;i<numberOfFish;i++){
			//randomly distributing fish within sea-cage
			alpha = randomGenerator.nextDouble()*pi*2;
			r0[0] = cageVector[0]+(float)Math.cos(alpha)*cageVector[2]*randomGenerator.nextFloat();
			r0[1] = cageVector[1]+(float)Math.sin(alpha)*cageVector[2]*randomGenerator.nextFloat();
			r0[2] = -cageVector[3]*randomGenerator.nextFloat();
			r0[3] = randomGenerator.nextFloat()*(float)Math.PI*2;
			r0[4] = randomGenerator.nextFloat()*(float)Math.PI*2;

			rOutTemp[i][0] = r0[0];
			rOutTemp[i][1] = r0[1];
			rOutTemp[i][2] = r0[2];

			//length weight relationship. SHOULD BE POSSIBLE TO GET BOTH FROM DATA
			float b0 = -19.8710f;							//Wing et al. 1998
			float b1 = 3.2123f;								//Wing et al. 1998
			
			if(fishSizeFormat == 'W'){

				BW = (float) (meanBodyWeightOrLength+weightOrLengthStandardDeviation*randomGenerator.nextGaussian());
				BL = (float)Math.pow((BW/1000)/Math.exp(b0),1/b1)/1000;
				fishList.add(new Fish(BW,BL,environment,r0,i,this));
			}
			else{
				BL = (float) (meanBodyWeightOrLength+weightOrLengthStandardDeviation*randomGenerator.nextGaussian());
				BW = (float) (Math.exp(b0)*Math.pow(BL*1000,b1))*1000;
				fishList.add(new Fish(BW,BL,environment,r0,i,this));

			}
			if(BL>maxBodyLength){
				maxBodyLength = BL;
			}
		}
		createBinLattice();
		System.out.println(numberOfFish+" fish created and randomly distributed in cage");
		
		System.out.println("Initialisation of model complete");
	}
	public void simulate(String scenarioName,int period,char treatment){
		//creating ComputationModules
		int numProcessors = Runtime.getRuntime().availableProcessors();	
		
		int fishNominalDivisionSize = (int)Math.floor((double)numberOfFish/(double)numProcessors);
		int fishLastDivisionSize = numberOfFish-fishNominalDivisionSize*(numProcessors-1)-1;
		long startTime;
		long endTime;
		ArrayList<ComputationModule> threadList = new ArrayList<ComputationModule>();
		
		for (int i = 0;i<numProcessors;i++){
			if(i==0){
				threadList.add(new ComputationModule(0,fishNominalDivisionSize,this,decimation,i));				
			}
			else if(i<numProcessors-1){
				threadList.add(new ComputationModule(i*fishNominalDivisionSize+1,(i+1)*fishNominalDivisionSize,this,decimation,i));
			}
			else{
				threadList.add(new ComputationModule(i*fishNominalDivisionSize+1,i*fishNominalDivisionSize+fishLastDivisionSize,this,decimation,i));
			}
		}
		
		System.out.println("Ouput arrays created, simulation commencing");
		startTime = System.currentTimeMillis();
		long time1 = startTime;
		long time2;
		long timeDiff;
		//simulating
		for (int i = 0;i<timeHorizon;i ++){
			if(i%10000 == 0){
				time2 = System.currentTimeMillis();
				timeDiff = (time2-time1);
				time1 = time2;
				System.out.println("Simulating time step "+i+". Millisec used since last output: "+timeDiff);
			}
			ArrayList <Thread> threadArray = new ArrayList<Thread>();
			findBinLatticeLocation(rOutTemp);
			
			for (int j=0;j<numProcessors;j++){
				threadList.get(j).setParams(i, timeStep, binLatticeLocations);
				Thread thread = new Thread(threadList.get(j));
				thread.start();
				threadArray.add(thread);
			}
			for (int j=0;j<numProcessors;j++){
				try {
					threadArray.get(j).join();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
			feedingFishProportion = numberOfFishFeeding/numberOfFish;
		}
		endTime = System.currentTimeMillis();
		long timeConsumed = (endTime-startTime)/1000;
		
		System.out.println("Simulation completed. File output commencing");
		
		//creating output vectors for file output 
		float[] scenarioParameters = {cageVector[0],cageVector[1],cageVector[3],cageVector[2],timeHorizon,timeStep,numberOfFish,decimation};
		float [] bodyLengths = new float[numberOfFish];
		float [] stomachVolumes = new float[numberOfFish];
		int totalPelletsEaten = 0;
		float [] pelletsWeightRatio = new float[numberOfFish];
		for (int i = 0;i<numberOfFish;i++){
			Fish fish = fishList.get(i);
			bodyLengths[i] = fish.getBodyLength();
			stomachVolumes[i] = fish.getStomachVolume();
			totalPelletsEaten+=fish.getPelletsEaten();
			pelletsWeightRatio[i] = (fish.getPelletsEaten())/fish.getBodyWeight();
		}
		System.out.println("Total pellets eaten by fish: "+totalPelletsEaten);
		
		
		//creating strings for filenames
		StringBuffer stringBuffer = new StringBuffer();
		stringBuffer.append("SALMON ");
		stringBuffer.append(scenarioName);
		stringBuffer.append(" period ");
		stringBuffer.append(Integer.toString(period));
		stringBuffer.append(", treatment ");
		stringBuffer.append(Character.toString(treatment));
		stringBuffer.append(" (");
		stringBuffer.append(Integer.toString(numberOfFish));
		stringBuffer.append(" fish over ");
		stringBuffer.append(Integer.toString(timeHorizon));
		stringBuffer.append(" seconds) ");
		String fileNameHeader = new String(stringBuffer);
		ToolBox.writePositionsAndParameters(fileNameHeader,r1Out, r2Out, r3Out, r4Out, r5Out, scenarioParameters, bodyLengths,decimation);
		ToolBox.writeSwimmingVelocities(fileNameHeader,rDot1Out,rDot2Out,rDot3Out);
		ToolBox.writeVerticalDistributionFile(ToolBox.findVerticalDistribution(r3Out,scenarioParameters[2],numberOfFish), fileNameHeader, scenarioParameters);
		ToolBox.writeStomachContents(fileNameHeader, stomachContents, stomachVolumes,scenarioParameters);
		System.out.println("Time used in simulation : "+timeConsumed+" seconds");
	}
	public void createBinLattice(){
		float cageRadius = cageVector[2];
		float cageDepth = cageVector[3];

		
		if(5*maxBodyLength>2){
			cellWidth = (float)Math.ceil(5*maxBodyLength);
		}
		else{
			cellWidth = 2;
		}
		horRange = (int)Math.ceil(cageRadius*2f/cellWidth);
		verRange = (int)Math.ceil(cageDepth/cellWidth);
		delta = (horRange*cellWidth-cageRadius*2)/2;
		
		binLattice = new ArrayList[horRange][horRange][verRange];
		for (int i = 0;i<horRange;i++){
			for (int j = 0;j<horRange;j++){
				for (int k = 0;k<verRange;k++){
					binLattice[i][j][k] = new ArrayList<Fish>();				
				}
			}
		}
		System.out.println("Bin-lattice structure successfully created");
	}
	public void findBinLatticeLocation(float[][] r){

		//flush bin-lattice
		for (int i = 0;i<horRange;i++){
			for (int j = 0;j<horRange;j++){
				for (int k = 0;k<verRange;k++){
					if(!binLattice[i][j][k].isEmpty()){
						binLattice[i][j][k].clear();
					}
				}
			}
		}
		//find new locations in bin-lattice
		float x;
		float y;
		float z;
		float xIndex;
		float yIndex;
		float zIndex;
		
		//find bin lattice locations
		for(int i = 0;i<numberOfFish;i++){
			float[] cageCentre = {cageVector[0],cageVector[1]};
			float cageRadius = cageVector[2];
			//float cageDepth = cageVector[3];
			
			x = r[i][0];
			y = r[i][1];
			z = r[i][2];
			
			zIndex = (float)Math.floor(-z);
			int tempIndex = 0;
			if(z<0){
				while(zIndex%cellWidth !=0){
					zIndex -=1;
					tempIndex++;
					if(tempIndex >1000){
						System.out.println("Deadlock in zIndex loop: "+zIndex+" "+z);
					}
				}
			}
			else {
				zIndex = 0;
			}
			zIndex = zIndex/cellWidth;
			

			xIndex = (float)Math.floor(x+cageRadius+delta-cageCentre[0]);
			tempIndex = 0;
			if(x>(float)Math.floor(x-cageRadius-delta-cageCentre[0])){
				while(xIndex%cellWidth !=0){
					xIndex -=1;
					if(tempIndex >1000){
						System.out.println("Deadlock in xIndex loop: "+xIndex);
					}
				}
			}
			else {
				xIndex = 0;
			}
			xIndex = xIndex/cellWidth;

			yIndex = (float)Math.floor(y+cageRadius+delta-cageCentre[1]);
			tempIndex = 0;
			if(y>(float)Math.floor(y-cageRadius-delta-cageCentre[1])){
				while(yIndex%cellWidth !=0){
					yIndex -=1;
					if(tempIndex >1000){
						System.out.println("Deadlock in yIndex loop: "+yIndex);
					}
				}
			}
			else{
				yIndex = 0;
			}
			yIndex = yIndex/cellWidth;
			
			//safeguard; making sure that the indexes are not outside bin-lattice
			if(xIndex>=horRange){
				xIndex = horRange-1;
			}
			if(xIndex<0){
				xIndex = 0;
			}
			if(yIndex>=horRange){
				yIndex = horRange-1;
			}
			if(yIndex<0){
				yIndex = 0;
			}
			if(zIndex>=verRange){
				zIndex = verRange-1;
			}
			if(zIndex<0){
				zIndex = 0;
			}
			
			int[] binLatticeLocation = {(int)xIndex,(int)yIndex,(int)zIndex}; 
			binLatticeLocations[i] = binLatticeLocation;
			binLattice[(int)xIndex][(int)yIndex][(int)zIndex].add(fishList.get(i));
		}
	}
	public void run(){
		initialiseModel();
		simulate(scenarioName,period,treatment);
		
	}
	public static void main(String[] args) {
		// initialising model
//		String scenarioName = "Johansson et al. 2006";
		String scenarioName = "Schooling Scenario";
		int period = 1;
		char treatment = 'N';
		int numberOfFish = 1;
		int timeHorizon = 86400;
		int timeStep = 1;
		int decimation = 1;
		Runtime runtime = Runtime.getRuntime();

		long totalMemory = runtime.totalMemory();
		long totalDataAmount = (long)numberOfFish*timeHorizon;
		while(totalDataAmount>totalMemory/2){
			decimation = decimation*2;
			totalDataAmount/=decimation;
		}
		decimation = 1;
		System.out.println("Total memory: "+totalMemory+", memory usage: "+totalDataAmount+", decimation: "+decimation);
		ArtiFish2_0Main mainThread = new ArtiFish2_0Main(numberOfFish,timeHorizon,timeStep,scenarioName,period,treatment,decimation);
		mainThread.start();
		System.out.println("Multi-threading commencing");
	}
}
