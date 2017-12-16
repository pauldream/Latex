#define _USE_MATH_DEFINES

#include <ilcplex/ilocplex.h>
#include <ilcplex/ilocplexi.h>
//#include <map>
#include <vector>
#include <list>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <queue>
#include <functional>
#include <time.h>
ILOSTLBEGIN

using namespace std;

typedef IloArray<IloNumVarArray> IloNumVarArray2;
typedef IloArray<IloNumVarArray2> IloNumVarArray3;
typedef IloArray<IloNumVarArray3> IloNumVarArray4;
typedef IloArray<IloIntVarArray> IloIntVarArray2;
typedef IloArray<IloNumArray> IloNumD2;
typedef IloArray<IloNumD2> IloNumD3;
typedef IloArray<IloNumD3> IloNumD4;

#define PI 3.14159265
const int NUM_CUS = 19;
const int NUM_FAC = 19 + 1;
const int NUM_STA = 31 + 0 + 1;
const int NUM_LEVEL = 15 + 0 + 1;
const int Num_Iteration = 20;
const float PENAL = 1000.0;	 //demand loss penalty
const int Count_Tree_Lim = 1200;//was 200000
const float timeLimit = 60;// 86400;
const float greedyZ_U = 2000000;


//common parameters
const float nu = 1.5f;		//step size control in LR*/
const float INFSMALL=0.000000001;
const int isHungarian=0;
float prob[NUM_STA] = {};
float fixed_cost[NUM_FAC] = {};
float demand[NUM_CUS] = {};
double lon[NUM_CUS] = {};
double lat[NUM_CUS] = {};
double beta[NUM_LEVEL] = { 1.0 }; // , 0.4, 0.1777, 0.0889, 0.0444, 0.0222, 0.011158, 0.0068, 0.004254, 0.002658, 0.00166, 0.001038, 0.00068, 0.000455, 0.0003};// {1, 0.12, 0.0192, 0.003072};

int Count_Tree=0;
float Z_U_Global=IloInfinity;
float Z_D_Global=0;
float beginTime;
float endTime;
bool *x_opt = new bool [NUM_FAC];

struct BBNode{
	bool* X_fixed;
	bool* X_mask;

    //lower bound
    float Z_D;
	float*** muS; //simplified mu
   
    //feasible solution
    float Z_U;

	int temp_j;
	friend bool operator > (const BBNode& lth, const BBNode& rth) {
		return lth.Z_D > rth.Z_D;
	}
};

struct ANode{
	int v;
	list<int> children;
	int parent;
};

priority_queue<BBNode,vector<BBNode>, greater<BBNode> > Live_queue;


// Function definition
int Eval_Problem(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,
			   const IloIntArray x_val, const IloInt print, float &totalcost);
void SolveLR(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int> > &JAss,const vector<vector<int> > &KAss,
			 bool* X, float &Z_D_new, float &Z_U_new, float ***mu, bool* X_fixed, bool* X_mask, int IterNum, float Lamb, float dec_ratio);
void OneStepGreedy(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss,
				   int &temp_j, float &cost_reduction,  bool* X, bool* X_mask, bool isInX);
void BB(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss,
		bool* X, float &Z_D_new, float &Z_U_new, float ***mu,  bool* X_fixed, bool* X_mask , int j_mask);
void SolveBB(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss,
			 int isBreadth);
void GetCost(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss, bool* X, float &Z_U);
void BBBFS(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss,
		bool* X, float ***mu, BBNode &thisBBNode);
void Augment(int k,int *exposed, int *label, int *mateRK,int *mateK);
int Modify(const int R_max,float *slack,float *alphaH, float *betaH,list<int>& Q,ANode *ATree,int *exposed, int *label,int *nhbor, int *mateRK,int *mateK);
void Hungarian(float &result, int temp_R,const vector<vector<int>> &link,int *mateRJ, int *mateRK,float ***coef_Y);
int Sub_Problem(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,
			   float ***lambda, const bool* x_fixed, const bool* x_mask, bool ****y_val, float &subcost,float &UB_sub2);
void PrintCost(const vector<vector<vector<float>>> & distance,const vector<vector<int> > &JAss,const vector<vector<int> > &KAss, 
			 bool* X, float &Z_U,bool print);



int main (int argc, char **argv)
{
	// read all parameters
	ifstream input_file;
	input_file.open("19-node-4local.txt", ios::in);
	string line;
	int count = 0;
	while (getline(input_file, line))
	{
		vector<double> line_component;
		stringstream ss(line);
		string tok;
		char delimiter = ' ';
		while (getline(ss, tok, delimiter))
		{
			line_component.push_back(atof(tok.c_str()));
		}
		fixed_cost[count] = line_component[5] / 1.0;
		demand[count] = line_component[3] / 100000.0;
		lon[count] = line_component[1] * PI / 180.0;
		lat[count] = line_component[2] * PI / 180.0;
		count++;
	}

	for (int k = 0; k<NUM_STA-1; k++)
	{
		prob[k] = 0.0;
	}
	prob[NUM_STA - 1] = 0.0;
	
	vector<vector<int>> link;
	link.resize(NUM_FAC);
	for (int j = 0; j<NUM_FAC; j++) {
		link[j].resize(NUM_STA);
		for (int k = 0; k<NUM_STA; k++) {
			link[j][k] = 0;
		}
	}
	link[NUM_FAC - 1][NUM_STA - 1] = 1;
	
	vector<vector<vector<float>>> distance;
	distance.resize(NUM_STA);
	for (int k = 0; k < NUM_STA; k++) {
		distance[k].resize(NUM_CUS);
		for (int i = 0; i < NUM_CUS; i++) {
			distance[k][i].resize(NUM_FAC);
			for (int j = 0; j < NUM_FAC; j++) {
				distance[k][i][j] = 100000;
			}
		}
	}
	for (int i = 0; i < NUM_CUS; i++) {
		for (int j = 0; j < NUM_FAC; j++) {
			double c = sin(lat[i]) * sin(lat[j]) + cos(lat[i]) * cos(lat[j]) * cos(-lon[i] + lon[j]);
			if (c > 1)
				c = 1;
			if (c < -1)
				c = -1;
			for (int k = 0; k < NUM_STA; k++) {
				distance[k][i][j] = floor(3959 * acos(c) + 0.5);
			}
		}
	}
	for (int i = 0; i<NUM_CUS; i++) {
		distance[NUM_STA - 1][i][NUM_FAC - 1] = PENAL;
	}


	// [14-node example]
	/*prob[0] = 0.5000;  prob[1] = 0.5000;  prob[2] = 0.8696;  prob[3] = 0.6250;  prob[4] = 0.9946;  prob[5] = 0.6250;  prob[6] = 0.9946;  prob[7] = 0.9758; prob[8] = 0.6100;
	prob[9] = 0.4000;  prob[10] = 0.4444; prob[11] = 0.9783; prob[12] = 0.6970; prob[13] = 0.6571; prob[14] = 0.5022;
	prob[15] = 0.5000; prob[16] = 0.6250; prob[17] = 0.6667; prob[18] = 0.7059; prob[19] = 0.6250; prob[20] = 0.8889; prob[21] = 0.9600; prob[22] = 0.9783; prob[23] = 0.6667; prob[24] = 0.9775;
	prob[NUM_STA - 1] = 0.0000;
	link[3][0] = 1; link[4][1] = 1; link[3][2] = 1; link[4][2] = 1; link[2][3] = 1; link[4][3] = 1; link[2][4] = 1; link[3][4] = 1; link[4][4] = 1;
	link[1][5] = 1; link[3][5] = 1; link[1][6] = 1; link[3][6] = 1; link[4][6] = 1; link[1][7] = 1; link[2][7] = 1; link[3][7] = 1; link[4][7] = 1;
	link[0][8] = 1; link[1][8] = 1; link[2][8] = 1; link[3][8] = 1; link[4][8] = 1;
	link[7][9] = 1; link[6][10] = 1; link[6][11] = 1; link[7][11] = 1; link[6][12] = 1; link[7][12] = 1; link[8][12] = 1;
	link[5][13] = 1; link[6][13] = 1; link[7][13] = 1; link[5][14] = 1; link[6][14] = 1; link[7][14] = 1; link[8][14] = 1;
	link[13][15] = 1; link[12][16] = 1; link[13][16] = 1; link[11][17] = 1; link[12][17] = 1; link[13][17] = 1;
	link[10][18] = 1; link[11][18] = 1; link[12][18] = 1; link[13][18] = 1; link[9][19] = 1; link[9][20] = 1; link[13][20] = 1;
	link[9][21] = 1; link[12][21] = 1; link[13][21] = 1; link[9][22] = 1; link[11][22] = 1; link[12][22] = 1; link[13][22] = 1;
	link[9][23] = 1; link[10][23] = 1; link[9][24] = 1; link[10][24] = 1; link[11][24] = 1; link[12][24] = 1; link[13][24] = 1;*/

	// [14-node 3 local]
	/*for (int j = 0; j < NUM_FAC; j++) {
	link[j][j] = 1;
	}*/
	/*prob[0] = 0.4286;  prob[1] = 0.6364;  prob[2] = 0.6364;  prob[3] = 0.8643;  prob[4] = 0.8000;  prob[5] = 0.5000;  
	prob[6] = 0.5000;  prob[7] = 0.5000; prob[8] = 0.8696; prob[9] = 0.6250;  prob[10] = 0.9946; prob[11] = 0.6250; 
	prob[12] = 0.9946; prob[13] = 0.9758; prob[14] = 0.6100;
	prob[15] = 0.4000; prob[16] = 0.4444; prob[17] = 0.9783; prob[18] = 0.6970; prob[19] = 0.6571; prob[20] = 0.5022; 
	prob[NUM_STA - 1] = 0.0000;
	link[0][0] = 1; link[0][1] = 1; link[2][1] = 1; link[0][2] = 1; link[1][2] = 1; link[0][3] = 1; link[1][3] = 1; link[2][3] = 1; 
	link[0][4] = 1; link[1][4] = 1; link[2][4] = 1; link[3][4] = 1; link[0][5] = 1; link[1][5] = 1; link[2][5] = 1; link[3][5] = 1;
	link[4][5] = 1; 
	link[8][6] = 1; link[9][7] = 1; link[8][8] = 1; link[9][8] = 1; link[7][9] = 1; link[9][9] = 1; link[7][10] = 1;
	link[8][10] = 1; link[9][10] = 1; link[6][11] = 1; link[8][11] = 1; link[6][12] = 1; link[8][12] = 1; link[9][12] = 1;
	link[6][13] = 1; link[7][13] = 1; link[8][13] = 1; link[9][13] = 1; link[5][14] = 1; link[6][14] = 1; link[7][14] = 1;
	link[8][14] = 1; link[9][14] = 1;
	link[12][15] = 1; link[11][16] = 1; link[11][17] = 1; link[12][17] = 1; link[11][18] = 1; link[12][18] = 1; link[13][18] = 1;
	link[10][19] = 1; link[11][19] = 1; link[12][19] = 1; link[10][20] = 1; link[11][20] = 1; link[12][20] = 1; link[13][20] = 1;*/

	// [17-node 3 local]
	/*for (int j = 0; j < NUM_FAC; j++) {
	link[j][j] = 1;
	}*/
	/*prob[0] = 0.4286;  prob[1] = 0.6364;  prob[2] = 0.6364;  prob[3] = 0.8643;  prob[4] = 0.8000;  prob[5] = 0.5000;
	prob[6] = 0.5000;  prob[7] = 0.5000; prob[8] = 0.8696; prob[9] = 0.6250;  prob[10] = 0.9946; prob[11] = 0.6250;
	prob[12] = 0.9946; prob[13] = 0.9758; prob[14] = 0.6100;
	prob[15] = 0.4000; prob[16] = 0.4444; prob[17] = 0.9783; prob[18] = 0.6970; prob[19] = 0.6571; prob[20] = 0.5022;
	for (int k = 21; k < NUM_STA - 1; k++) {
		prob[k] = 0.1500;
	}
	prob[NUM_STA - 1] = 0.0000;
	link[0][0] = 1; link[0][1] = 1; link[2][1] = 1; link[0][2] = 1; link[1][2] = 1; link[0][3] = 1; link[1][3] = 1; link[2][3] = 1;
	link[0][4] = 1; link[1][4] = 1; link[2][4] = 1; link[3][4] = 1; link[0][5] = 1; link[1][5] = 1; link[2][5] = 1; link[3][5] = 1;
	link[4][5] = 1;
	link[10][6] = 1; link[11][7] = 1; link[10][8] = 1; link[11][8] = 1; link[9][9] = 1; link[11][9] = 1; link[9][10] = 1;
	link[10][10] = 1; link[11][10] = 1; link[8][11] = 1; link[10][11] = 1; link[8][12] = 1; link[10][12] = 1; link[11][12] = 1;
	link[8][13] = 1; link[9][13] = 1; link[10][13] = 1; link[11][13] = 1; link[7][14] = 1; link[8][14] = 1; link[9][14] = 1;
	link[10][14] = 1; link[11][14] = 1;
	link[14][15] = 1; link[13][16] = 1; link[13][17] = 1; link[14][17] = 1; link[13][18] = 1; link[14][18] = 1; link[15][18] = 1;
	link[12][19] = 1; link[13][19] = 1; link[14][19] = 1; link[12][20] = 1; link[13][20] = 1; link[14][20] = 1; link[15][20] = 1; 
	
	link[5][21] = 1; link[6][22] = 1; link[16][23] = 1;*/

	// [19-node 4 local]
	/*for (int j = 0; j < NUM_FAC; j++) {
		link[j][j] = 1;
		if(j < NUM_FAC - 1)
			prob[j] = 0.05 + (j%4)*0.05;
	}*/
	prob[0] = 0.4286;  prob[1] = 0.6364;  prob[2] = 0.6364;  prob[3] = 0.8643;  prob[4] = 0.8000;  prob[5] = 0.5000;
	prob[6] = 0.5000;  prob[7] = 0.5000;  prob[8] = 0.8696;  prob[9] = 0.6250;  prob[10] = 0.9946;  prob[11] = 0.6250;  
	prob[12] = 0.9946;  prob[13] = 0.9758; prob[14] = 0.6100;
	prob[15] = 0.4000;  prob[16] = 0.4444; prob[17] = 0.9783; prob[18] = 0.6970; prob[19] = 0.6571; prob[20] = 0.5022;
	prob[21] = 0.5000; prob[22] = 0.6250; prob[23] = 0.6667; prob[24] = 0.7059; prob[25] = 0.6250; prob[26] = 0.8889; 
	prob[27] = 0.9600; prob[28] = 0.9783; prob[29] = 0.6667; prob[30] = 0.9775;
	prob[NUM_STA - 1] = 0.0000;
	link[0][0] = 1; link[0][1] = 1; link[2][1] = 1; link[0][2] = 1; link[1][2] = 1; link[0][3] = 1; link[1][3] = 1; 
	link[2][3] = 1; link[0][4] = 1; link[1][4] = 1; link[2][4] = 1; link[3][4] = 1; link[0][5] = 1; link[1][5] = 1; 
	link[2][5] = 1; link[3][5] = 1; link[4][5] = 1;
	link[8][6] = 1; link[9][7] = 1; link[8][8] = 1; link[9][8] = 1; link[7][9] = 1; link[9][9] = 1; link[7][10] = 1; 
	link[8][10] = 1; link[9][10] = 1; link[6][11] = 1; link[8][11] = 1; link[6][12] = 1; link[8][12] = 1; link[9][12] = 1; 
	link[6][13] = 1; link[7][13] = 1; link[8][13] = 1; link[9][13] = 1;
	link[5][14] = 1; link[6][14] = 1; link[7][14] = 1; link[8][14] = 1; link[9][14] = 1;
	link[12][15] = 1; link[11][16] = 1; link[11][17] = 1; link[12][17] = 1; link[11][18] = 1; link[12][18] = 1; link[13][18] = 1;
	link[10][19] = 1; link[11][19] = 1; link[12][19] = 1; link[10][20] = 1; link[11][20] = 1; link[12][20] = 1; link[13][20] = 1;
	link[18][21] = 1; link[17][22] = 1; link[18][22] = 1; link[16][23] = 1; link[17][23] = 1; link[18][23] = 1; 
	link[15][24] = 1; link[16][24] = 1; link[17][24] = 1; link[18][24] = 1; link[14][25] = 1; link[14][26] = 1; 
	link[18][26] = 1; link[14][27] = 1; link[17][27] = 1; link[18][27] = 1; 
	link[14][28] = 1; link[16][28] = 1; link[17][28] = 1; link[18][28] = 1; 
	link[14][29] = 1; link[15][29] = 1; 
	link[14][30] = 1; link[15][30] = 1; link[16][30] = 1; link[17][30] = 1; link[18][30] = 1;
	
	// [25-node 4 local]
	/*for (int j = 0; j < NUM_FAC; j++) {
		link[j][j] = 1;
	}*/	
	//prob[0] = 0.4286;  prob[1] = 0.6364;  prob[2] = 0.6364;  prob[3] = 0.8643;  prob[4] = 0.8000;  prob[5] = 0.5000;
	//prob[6] = 0.5000;  prob[7] = 0.5000;  prob[8] = 0.8696;  prob[9] = 0.6250;  prob[10] = 0.9946;  prob[11] = 0.6250;
	//prob[12] = 0.9946;  prob[13] = 0.9758; prob[14] = 0.6100;
	//prob[15] = 0.4000;  prob[16] = 0.4444; prob[17] = 0.9783; prob[18] = 0.6970; prob[19] = 0.6571; prob[20] = 0.5022;
	//prob[21] = 0.5000; prob[22] = 0.6250; prob[23] = 0.6667; prob[24] = 0.7059; prob[25] = 0.6250; prob[26] = 0.8889;
	//prob[27] = 0.9600; prob[28] = 0.9783; prob[29] = 0.6667; prob[30] = 0.9775;
	//for (int k = 31; k < NUM_STA - 1; k++) {
	//	prob[k] = 0.0000;
	//}
	//prob[NUM_STA - 1] = 0.0000;
	//link[0][0] = 1; link[0][1] = 1; link[2][1] = 1; link[0][2] = 1; link[1][2] = 1; link[0][3] = 1; link[1][3] = 1;
	//link[2][3] = 1; link[0][4] = 1; link[1][4] = 1; link[2][4] = 1; link[3][4] = 1; link[0][5] = 1; link[1][5] = 1;
	//link[2][5] = 1; link[3][5] = 1; link[4][5] = 1; // local 1
	//link[10][6] = 1; link[11][7] = 1; link[10][8] = 1; link[11][8] = 1; link[9][9] = 1; link[11][9] = 1; link[9][10] = 1;
	//link[10][10] = 1; link[11][10] = 1; link[8][11] = 1; link[10][11] = 1; link[8][12] = 1; link[10][12] = 1; link[11][12] = 1;
	//link[8][13] = 1; link[9][13] = 1; link[10][13] = 1; link[11][13] = 1;
	//link[7][14] = 1; link[8][14] = 1; link[9][14] = 1; link[10][14] = 1; link[11][14] = 1; // local 2
	//link[14][15] = 1; link[13][16] = 1; link[13][17] = 1; link[14][17] = 1; link[13][18] = 1; link[14][18] = 1; link[15][18] = 1;
	//link[12][19] = 1; link[13][19] = 1; link[14][19] = 1; link[12][20] = 1; link[13][20] = 1; link[14][20] = 1; link[15][20] = 1; // local 3
	//link[24][21] = 1; link[23][22] = 1; link[24][22] = 1; link[22][23] = 1; link[23][23] = 1; link[24][23] = 1;
	//link[21][24] = 1; link[22][24] = 1; link[23][24] = 1; link[24][24] = 1; link[20][25] = 1; link[20][26] = 1;
	//link[24][26] = 1; link[20][27] = 1; link[23][27] = 1; link[24][27] = 1;
	//link[20][28] = 1; link[22][28] = 1; link[23][28] = 1; link[24][28] = 1;
	//link[20][29] = 1; link[21][29] = 1;
	//link[20][30] = 1; link[21][30] = 1; link[22][30] = 1; link[23][30] = 1; link[24][30] = 1; // local 4
	//link[5][31] = 1; link[6][32] = 1; link[16][33] = 1; link[17][34] = 1; link[18][35] = 1; link[19][36] = 1; // other locations
	

	// [49-node 8 local]
	/*for (int j = 0; j < NUM_FAC; j++) {
		link[j][j] = 1;
	}
	float lonNO = 89.93136 * PI / 180.0;
	float latNO = 30.06585 * PI / 180.0;
	double distance_temp[NUM_FAC - 1] = {};
	for (int j = 0; j < NUM_FAC - 1; j++) {
		double c = sin(latNO) * sin(lat[j]) + cos(latNO) * cos(lat[j]) * cos(-lonNO + lon[j]);
		if (c > 1)
			c = 1;
		if (c < -1)
			c = -1;
		distance_temp[j] = floor(3959 * acos(c) + 0.5);
		prob[j] = 0.1 * exp(-1.0 * distance_temp[j] / 500.0);
	}*/
	/*
	// local 1
	prob[0] = 0.4000;  prob[1] = 0.5556;  prob[2] = 0.4000;  prob[3] = 0.9615;  prob[4] = 1.0636;  prob[5] = 0.5556;
	prob[6] = 1.0636;  prob[7] = 1.0062; prob[8] = 0.8222; prob[9] = 0.8222;  prob[10] = 0.5473; 
	link[6][0] = 1; link[5][1] = 1; link[6][1] = 1; link[4][2] = 1; link[4][3] = 1; link[6][3] = 1; link[4][4] = 1; 
	link[5][4] = 1; link[6][4] = 1; link[3][5] = 1; link[4][5] = 1; link[3][6] = 1; link[4][6] = 1; link[6][6] = 1;
	link[3][7] = 1; link[4][7] = 1; link[5][7] = 1; link[6][7] = 1; link[2][8] = 1; link[3][8] = 1; link[4][8] = 1;
	link[5][8] = 1; link[6][8] = 1; link[1][9] = 1; link[3][9] = 1; link[4][9] = 1; link[5][9] = 1; link[6][9] = 1;
	link[1][10] = 1; link[2][10] = 1; link[3][10] = 1; link[4][10] = 1; link[5][10] = 1; link[6][10] = 1;
	// local 2
	prob[11] = 0.4286; prob[12] = 0.6364; prob[13] = 0.6364; prob[14] = 0.8643; prob[15] = 0.8000; prob[16] = 0.5000;
	link[7][11] = 1; link[7][12] = 1; link[9][12] = 1; link[7][13] = 1; link[8][13] = 1; link[7][14] = 1; link[8][14] = 1;
	link[9][14] = 1; link[7][15] = 1; link[8][15] = 1; link[9][15] = 1; link[10][15] = 1; link[7][16] = 1; link[8][16] = 1;
	link[9][16] = 1; link[10][16] = 1; link[11][16] = 1;
	// local 3
	prob[17] = 0.5000; prob[18] = 0.5000; prob[19] = 0.8696; prob[20] = 0.6250; prob[21] = 0.9946; prob[22] = 0.6250;
	prob[23] = 0.9946; prob[24] = 0.9758; prob[25] = 0.6100; 
	link[17][17] = 1; link[18][18] = 1; link[17][19] = 1; link[18][19] = 1; link[16][20] = 1; link[18][20] = 1; 
	link[16][21] = 1; link[17][21] = 1; link[18][21] = 1; link[15][22] = 1; link[17][22] = 1; link[15][23] = 1; 
	link[17][23] = 1; link[18][23] = 1; link[15][24] = 1; link[16][24] = 1; link[17][24] = 1; link[18][24] = 1;
	link[14][25] = 1; link[15][25] = 1; link[16][25] = 1; link[17][25] = 1; link[18][25] = 1;
	// local 4
	prob[26] = 0.4000; prob[27] = 0.4444; prob[28] = 0.9783; prob[29] = 0.6970; prob[30] = 0.6571; prob[31] = 0.5022;
	link[21][26] = 1; link[20][27] = 1; link[20][28] = 1; link[21][28] = 1; link[20][29] = 1; link[21][29] = 1; 
	link[22][29] = 1; link[19][30] = 1; link[20][30] = 1; link[21][30] = 1; link[19][31] = 1; link[20][31] = 1; 
	link[21][31] = 1; link[22][31] = 1;
	// local 5
	prob[32] = 0.5000; prob[33] = 0.6250; prob[34] = 0.6667; prob[35] = 0.7059; prob[36] = 0.6250; prob[37] = 0.8889;
	prob[38] = 0.9600; prob[39] = 0.9783; prob[40] = 0.6667; prob[41] = 0.9775;
	link[31][32] = 1; link[30][33] = 1; link[31][33] = 1; link[29][34] = 1; link[30][34] = 1; link[31][34] = 1;
	link[28][35] = 1; link[29][35] = 1; link[30][35] = 1; link[31][35] = 1; link[27][36] = 1; link[27][37] = 1;
	link[31][37] = 1; link[27][38] = 1; link[30][38] = 1; link[31][38] = 1;
	link[27][39] = 1; link[29][39] = 1; link[30][39] = 1; link[31][39] = 1;
	link[27][40] = 1; link[28][40] = 1;
	link[27][41] = 1; link[28][41] = 1; link[29][41] = 1; link[30][41] = 1; link[31][41] = 1;
	// local 6
	prob[42] = 0.4000; prob[43] = 0.5556; prob[44] = 0.6429; prob[45] = 0.4444; prob[46] = 1.0714; prob[47] = 1.0216;
	prob[48] = 0.6429; prob[49] = 1.0208; prob[50] = 0.7368; prob[51] = 1.1903;
	link[35][42] = 1; link[34][43] = 1; link[35][43] = 1; link[33][44] = 1; link[34][44] = 1; link[35][44] = 1;
	link[32][45] = 1; link[32][46] = 1; link[35][46] = 1; link[32][47] = 1; link[34][47] = 1; link[35][47] = 1;
	link[32][48] = 1; link[33][48] = 1; link[32][49] = 1; link[33][49] = 1; link[35][49] = 1;
	link[32][50] = 1; link[33][50] = 1; link[34][50] = 1;
	link[32][51] = 1; link[33][51] = 1; link[34][51] = 1; link[35][51] = 1;
	// local 7
	prob[52] = 0.5000; prob[53] = 0.5000; prob[54] = 0.4000;
	link[38][52] = 1; link[39][53] = 1; link[38][54] = 1; link[39][54] = 1;
	// local 8
	prob[55] = 0.5000; prob[56] = 0.5000; prob[57] = 0.3333; prob[58] = 0.8000; prob[59] = 0.8000; prob[60] = 0.9375;
	link[45][55] = 1; link[45][56] = 1; link[46][56] = 1; link[47][57] = 1; link[48][57] = 1;
	link[45][58] = 1; link[46][58] = 1; link[47][58] = 1; link[45][59] = 1; link[46][59] = 1; link[48][59] = 1;
	link[45][60] = 1; link[46][60] = 1; link[47][60] = 1; link[48][60] = 1;

	for (int k = 61; k < NUM_STA - 1; k++) {
		prob[k] = 0.1000;
	}
	prob[NUM_STA - 1] = 0.0000;
	link[0][61] = 1; link[12][62] = 1; link[13][63] = 1; link[23][64] = 1; link[24][65] = 1;
	link[25][66] = 1; link[36][67] = 1; link[37][68] = 1; link[40][69] = 1; link[41][70] = 1; 
	link[42][71] = 1; link[43][72] = 1; link[44][73] = 1;
	*/

	// [49-node simple]
	/*for (int j = 0; j < NUM_FAC; j++) {
	link[j][j] = 1;
	}*/
	/*prob[0] = 0.4286;  prob[1] = 0.6364;  prob[2] = 0.6364;  prob[3] = 0.8643;  prob[4] = 0.8000;  prob[5] = 0.5000;  
	prob[6] = 0.4000;  prob[7] = 0.4444; prob[8] = 0.9783; prob[9] = 0.6970;  prob[10] = 0.6571; prob[11] = 0.5022; 
	prob[12] = 0.5000; prob[13] = 0.6250; prob[14] = 0.6667; prob[15] = 0.7059; prob[16] = 0.6250; prob[17] = 0.8889; prob[18] = 0.9600; prob[19] = 0.9783; prob[20] = 0.6667; prob[21] = 0.9775; 
	for (int k = 22; k < NUM_STA - 1; k++) {
		prob[k] = 0.1500;
	}
	prob[NUM_STA - 1] = 0.0000;
	link[7][0] = 1; link[7][1] = 1; link[9][1] = 1; link[7][2] = 1; link[8][2] = 1; link[7][3] = 1; link[8][3] = 1; link[9][3] = 1;
	link[7][4] = 1; link[8][4] = 1; link[9][4] = 1; link[10][4] = 1; link[7][5] = 1; link[8][5] = 1; link[9][5] = 1; link[10][5] = 1; link[11][5] = 1;
	
	link[21][6] = 1; link[20][7] = 1; link[20][8] = 1; link[21][8] = 1; link[20][9] = 1; link[21][9] = 1; link[22][9] = 1;
	link[19][10] = 1; link[20][10] = 1; link[21][10] = 1; link[19][11] = 1; link[20][11] = 1; link[21][11] = 1; link[22][11] = 1;

	link[31][12] = 1; link[30][13] = 1; link[31][13] = 1; link[29][14] = 1; link[30][14] = 1; link[31][14] = 1;
	link[28][15] = 1; link[29][15] = 1; link[30][15] = 1; link[31][15] = 1; link[27][16] = 1; link[27][17] = 1; link[31][17] = 1;
	link[27][18] = 1; link[30][18] = 1; link[31][18] = 1; link[27][19] = 1; link[29][19] = 1; link[30][19] = 1; link[31][19] = 1;
	link[27][20] = 1; link[28][20] = 1; link[27][21] = 1; link[28][21] = 1; link[29][21] = 1; link[30][21] = 1; link[31][21] = 1;

	int stationIdx = 22;
	for (int j = 0; j < 7; j++) {
		link[j][stationIdx] = 1;
		stationIdx += 1;
	}
	for (int j = 12; j < 19; ++j) {
		link[j][stationIdx] = 1;
		stationIdx += 1;
	}
	for (int j = 23; j < 27; ++j) {
		link[j][stationIdx] = 1;
		stationIdx += 1;
	}
	for (int j = 32; j < NUM_FAC; ++j) {
		link[j][stationIdx] = 1;
		stationIdx += 1;
	}*/

	// [88-node]
	/*for (int j = 0; j < NUM_FAC; j++) {
		link[j][j] = 1;
	}
	for (int j = 0; j < NUM_FAC-1; j++) {
		prob[j] = 0.1 * exp(-distance[0][j][23] / 400.0);
	}*/


	// calculate lower approximation of [beta_r]
	const int c = 31;
	float probtemp[c] = {};
	for (int i = 0; i < c; ++i) {
		probtemp[i] = prob[i];
	}
	for (int i = 0; i < c - 1; ++i) {
		for (int j = 0; j < c - 1 - i; ++j) {
			if (probtemp[j] > probtemp[j + 1]) {
				float temp = probtemp[j];
				probtemp[j] = probtemp[j + 1];
				probtemp[j+1] = temp;
			}
		}
	}
	for (int r = 1; r < NUM_LEVEL; ++r) {
		beta[r] = beta[r - 1] * probtemp[r - 1];
		cout << beta[r - 1] << " * " << probtemp[r-1] << " = " << beta[r] << endl;
	}

	// cout << "NUM_LEVEL = " << NUM_LEVEL <<". Fixcost = "<<fixed_cost[0]<<". Network: "<<NUM_LENGTH<<"x"<<NUM_LENGTH<<"."<<endl;
	for(int j=0;j<NUM_FAC;j++) 
		x_opt[j]=0;
	x_opt[NUM_FAC-1]=1;

	// order the distance for each i and level r, and define JAss and KAss
	int i,j,k,r;
	float *** d_copy=new float **[NUM_CUS];
	for (i = 0; i < NUM_CUS; i++) {
		d_copy[i] = new float *[NUM_FAC];
		for (j = 0; j < NUM_FAC; j++) {
			d_copy[i][j] = new float[NUM_STA];
			for (k = 0; k < NUM_STA; k++) {
				d_copy[i][j][k] = distance[k][i][j];
			}
		}
	}

	vector<vector<int>> JAss;
	vector<vector<int>> KAss;
	JAss.resize(NUM_CUS);
	KAss.resize(NUM_CUS);
	float temp_d;
	int temp_j, temp_k;
	for (i = 0; i < NUM_CUS; i++)
	{
		JAss[i].clear();
		KAss[i].clear();
		for (r = 0; r < (NUM_FAC - 1)*(NUM_STA - 1); r++) {
			temp_j = -1;
			temp_k = -1;
			temp_d = numeric_limits<float>::infinity();
			for (j = 0; j < NUM_FAC - 1; j++) {
				for (k = 0; k < NUM_STA - 1; k++) {
					if (temp_d > d_copy[i][j][k] && link[j][k] == 1) { //link[j][k]&&
						temp_d = d_copy[i][j][k];
						temp_j = j;
						temp_k = k;
					}
				}
			}
			if (temp_j == -1 || d_copy[i][temp_j][temp_k] >= PENAL) { // if the current facility is worse than the emergency one
				break;
			}
			JAss[i].push_back(temp_j);
			KAss[i].push_back(temp_k);
			d_copy[i][temp_j][temp_k] = numeric_limits<float>::infinity();//a station can only serve one level
		}
		JAss[i].push_back(NUM_FAC - 1);
		KAss[i].push_back(NUM_STA - 1);
	}
	for (i = 0; i < NUM_CUS; i++) {
		delete[] d_copy[i];
	}
	delete[] d_copy;
	

	try {
		beginTime = (float)clock();
		SolveBB(distance, link, JAss, KAss, 1); // isBreadth = 1
		getchar();					
	}
	catch (IloException& e) {
		cerr << "Concert exception caught: " << e << endl;
	}
	catch (...) {
		cerr << "Unknown exception caught" << endl;
	}
	delete []x_opt;
	return 0;
}



void SolveBB(const vector<vector<vector<float>>> &distance, const vector<vector<int>> &link, const vector<vector<int>> &JAss, const vector<vector<int>> &KAss,
	int isBreadth) {

	int i, j, k, l, num_path;
	bool * X = new bool[NUM_FAC]; //emergency facility location
	bool * X_fixed = new bool[NUM_FAC];
	bool * X_mask = new bool[NUM_FAC];
	for (j = 0; j<NUM_FAC - 1; j++) 
	{ 
		X[j] = X_fixed[j] = X_mask[j] = 0; 
	}
	X[NUM_FAC - 1] = X_fixed[NUM_FAC - 1] = X_mask[NUM_FAC - 1] = 1;
	float BBTime;
	float Z_D_new, Z_U_new;

	// initialize multiplies
	Z_U_new = greedyZ_U;
	Z_D_new = -numeric_limits<float>::infinity();
	float *** mu = new float ** [NUM_CUS];  //define lagrangian multiplier
	for (i = 0; i < NUM_CUS; i++) {				 //initialize 
		mu[i] = new float *[NUM_FAC - 1];
		for (j = 0; j < NUM_FAC - 1; j++) {
			num_path = 0;
			mu[i][j] = new float[NUM_STA - 1];
			for (int k = 0; k < NUM_STA - 1; k++) {
				num_path += link[j][k];
			}
			for (int k = 0; k < NUM_STA - 1; k++) {
				if (link[j][k]) {
					mu[i][j][k] = fixed_cost[j] / NUM_CUS / num_path;
				}
				else {
					mu[i][j][k] = 0;
				}
			}
		}
	}

	SolveLR(distance, link, JAss, KAss, X, Z_D_new, Z_U_new, mu, X_fixed, X_mask, 2000, 1.0, 1.02f);	//was 2000 for the large case// lambda was 1.0

	float temp = Z_U_Global = Z_U_new;
	Z_D_Global = Z_D_new;

	for (j = 0; j < NUM_FAC; j++) {
		x_opt[j] = X[j];
	}
	cout << "First step: " << endl;
	cout << "LR BB objective value: " << Z_U_Global << " LB:" << Z_D_Global << endl;

	cout << "Facilities:\t";
	for (j = 0; j < NUM_FAC; j++) {
		if (x_opt[j]) {
			cout << j + 1 << "\t";
		}
	}
	cout << endl << endl;

	if (!isBreadth) {
		BB(distance, link, JAss, KAss, X, Z_D_new, Z_U_new, mu, X_fixed, X_mask, -1);
		Z_D_Global = Z_D_new>Z_D_Global ? Z_D_new : Z_D_Global;
		if (temp>Z_U_new) {
			Z_U_Global = Z_U_new;
			for (j = 0; j < NUM_FAC - 1; j++) {
				x_opt[j] = X[j];
			}
		}
		cout << "LR BB depth objective value: " << Z_U_Global << endl;
		cout << "LR BB LowerBound: " << Z_D_Global << endl;
		cout << "Facilities:\t";
		for (j = 0; j<NUM_FAC; j++)if (x_opt[j]) { //was X[j]
			cout << j + 1 << " " << endl;
		}
	}
	else {
		BBNode thisBBNode;
		thisBBNode.muS = new float **[NUM_CUS];
		for (i = 0; i < NUM_CUS; i++) {				 //initialize 
			thisBBNode.muS[i] = new float *[NUM_FAC - 1];
			for (j = 0; j < NUM_FAC - 1; j++) {
				num_path = 0;
				for (k = 0; k < NUM_STA - 1; k++)
					if (link[j][k])
						num_path++;
				thisBBNode.muS[i][j] = new float[num_path];
				l = 0;
				for (k = 0; k < NUM_STA - 1; k++) {
					if (link[j][k]) {
						thisBBNode.muS[i][j][l] = mu[i][j][k];
						l++;
					}
				}
			}
		}

		thisBBNode.X_fixed = new bool[NUM_FAC];
		thisBBNode.X_mask = new bool[NUM_FAC];
		for (j = 0; j<NUM_FAC; j++)
			thisBBNode.X_fixed[j] = thisBBNode.X_mask[j] = 0;
		thisBBNode.X_fixed[NUM_FAC - 1] = thisBBNode.X_mask[NUM_FAC - 1] = 1;
		thisBBNode.Z_D = Z_D_new;
		thisBBNode.Z_U = Z_U_new;
		thisBBNode.temp_j = -1;
		Live_queue.push(thisBBNode);
		Z_D_Global = numeric_limits<float>::infinity();

		while (Live_queue.size()>0) {
			if (Count_Tree % 100 == 0) { //was 10000 || Count_Tree> 130000
				endTime = (float)clock();
				BBTime = (endTime - beginTime) / CLOCKS_PER_SEC;
				cout << "count_Tree=" << Count_Tree << "\t" << "time=" << BBTime << "\t" << "UB=" << Z_U_Global << "\t" << "temp_LB=" << Live_queue.top().Z_D << endl;
				cout << "Facilities:\t";
				for (j = 0; j < NUM_FAC; j++) {
					if (x_opt[j]) {
						cout << j + 1 << "\t";
					}
				}
				cout << endl << endl;
			}

			BBNode node_temp = Live_queue.top();
			BBBFS(distance, link, JAss, KAss, X, mu, Live_queue.top()); //Live_queue.top()

			for (i = 0; i<NUM_CUS; i++) {
				for (j = 0; j<NUM_FAC - 1; j++) 
					delete[] Live_queue.top().muS[i][j];
				delete[] Live_queue.top().muS[i];
			}
			delete[] Live_queue.top().muS;
			delete[] Live_queue.top().X_fixed;
			delete[] Live_queue.top().X_mask;
			Live_queue.pop();
		}

		cout << "LR BB breadth objective value: " << Z_U_Global << endl;
		cout << "LR BB LowerBound: " << Z_D_Global << endl;
		cout << "Facilities:\t";
		for (j = 0; j<NUM_FAC; j++) {
			if (x_opt[j]) { //was X_opt[j]
				cout << j + 1 << " ";
			}
		}
		cout << endl;
	}

	Count_Tree = 0;
	IloEnv env_BB;
	IloIntArray X_eval(env_BB, NUM_FAC);
	for (j = 0; j < NUM_FAC; j++) X_eval[j] = x_opt[j];
	float cost_UB_global;

	endTime = (float)clock();
	BBTime = (endTime - beginTime) / CLOCKS_PER_SEC;
	cout << "Solution Time: " << BBTime << endl;

	PrintCost(distance, JAss, KAss, x_opt, cost_UB_global, 1);
	// Eval_Problem(distance, link, X_eval, 1, cost_UB_global);
	env_BB.end();

	for (i = 0; i<NUM_CUS; i++) {
		for (j = 0; j<NUM_FAC - 1; j++) delete[] mu[i][j];
		delete[] mu[i];
	}
	delete[] mu;
	delete[] X;
	delete[] X_fixed;
	delete[] X_mask;
}



void SolveLR(const vector<vector<vector<float>>> & distance, const vector<vector<int>> &link, const vector<vector<int>> &JAss, const vector<vector<int>> &KAss,
	bool* X, float &Z_D_new, float &Z_U_new, float ***mu, bool* X_fixed, bool* X_mask, int IterNum, float Lamb, float dec_ratio) {
	int i, j, k, r, r1, r_temp, temp_j, temp_k, flag;	//X is the optimal one, X_fixed is the value of x, X_mask identify which x is branched
	
	float Z_D, Z_U, Z_D_All = Z_D_new, Z_U_All = Z_U_new; //Z_D local lower bound; Z_U local upper bound;
	float temp_sum, temp, temp1, tempxx;
	int *mateRJ = new int[NUM_LEVEL];
	int *mateRK = new int[NUM_LEVEL];

	float * min_coef_X = new float[NUM_LEVEL];
	int * min_Jindex = new int[NUM_LEVEL];
	int * min_Kindex = new int[NUM_LEVEL];
	bool * X_temp = new bool[NUM_FAC];
	X_temp[NUM_FAC - 1] = 1;

	bool ****Y = new bool ***[NUM_CUS];
	for (i = 0; i < NUM_CUS; i++) {
		Y[i] = new bool **[NUM_FAC];
		for (j = 0; j < NUM_FAC; j++) {
			Y[i][j] = new bool *[NUM_STA];
			for (k = 0; k < NUM_STA; k++) {
				Y[i][j][k] = new bool[NUM_LEVEL];
				for (r = 0; r < NUM_LEVEL; r++) {
					Y[i][j][k][r] = 0;
				}
			}
		}
	}
	float ***coef_Y = new float **[NUM_FAC];
	for (j = 0; j < NUM_FAC; j++) {
		coef_Y[j] = new float *[NUM_STA];
		for (k = 0; k < NUM_STA; k++) {
			coef_Y[j][k] = new float[NUM_LEVEL];
			for (r = 0; r < NUM_LEVEL; r++) {
				coef_Y[j][k][r] = numeric_limits<float>::infinity();
			}
		}
	}


	// LR procedure
	int Count = 0; //IsCount
	int Count_1 = 0;
	int convergenceCount = 0;//DownCount=0, IsDownCount, 
	float worst_Z_D = numeric_limits<float>::infinity(), best_Z_D = -numeric_limits<float>::infinity();
	float den, num, tk;
	//for(j=0;j<J;j++) {Count_mask+=X_mask[j];Count_fixed+=X_fixed[j]*X_mask[j];}

	for (int Iteration = 0; Iteration < IterNum; Iteration++) {
		// lower bound
		Z_D = 0;
		//contribution of X
		for (j = 0; j < NUM_FAC - 1; j++) {
			if (X_temp[j] = X_fixed[j] & X_mask[j]) {// first assign x-temp with the intersection of fixed&mask, then check if x_temp==1
				temp_sum = fixed_cost[j];
				Z_D += temp_sum;
			}
		}
		for (j = 0; j < NUM_FAC - 1; j++) {
			if (!X_mask[j]) {//temp_sum is the mu_sum
				temp_sum = fixed_cost[j];
				for (i = 0; i < NUM_CUS; i++) {
					for (k = 0; k < NUM_STA - 1; k++) {
						if (link[j][k]) {
							temp_sum -= mu[i][j][k];
						}
					}
				}
				if (temp_sum < 0) {
					X_temp[j] = 1;
					Z_D += temp_sum;
				}
			}
		}

		//contribution of Y
		for (i = 0; i < NUM_CUS; i++) {
			if (!isHungarian) {//not using hungarian //!isHungarian
				for (r = 0; r < NUM_LEVEL - 1; r++) {
					for (j = 0; j < NUM_FAC; j++) {
						for (k = 0; k < NUM_STA; k++) {
							if (link[j][k]) {
								Y[i][j][k][r] = 0;
								coef_Y[j][k][r] = numeric_limits<float>::infinity();
								if (X_fixed[j] && X_mask[j])
								{
									coef_Y[j][k][r] = demand[i] * distance[k][i][j] * (1 - prob[k])*beta[r];
								}
								if ((!X_fixed[j]) && (!X_mask[j]))
								{
									coef_Y[j][k][r] = demand[i] * distance[k][i][j] * (1 - prob[k])*beta[r] + mu[i][j][k];
								}
							}
						}
					}
				}
				coef_Y[NUM_FAC - 1][NUM_STA - 1][NUM_LEVEL - 1] = demand[i] * PENAL * beta[NUM_LEVEL - 1];
				temp_k = temp_j = -1;

				//find the lowest cost coefficient for each Yijkr
				for (r = 0; r < NUM_LEVEL - 1; r++) {
					min_coef_X[r] = numeric_limits<float>::infinity();
					for (k = 0; k < NUM_STA - 1; k++) {
						flag = 0;
						for (int r1 = 0; r1 < r; r1++) {
							if (k == min_Kindex[r1]) {
								flag = 1; 
								break;
							} //a station can only serve one level
						}
						if (flag) continue;
						for (j = 0; j < NUM_FAC - 1; j++) {
							if (link[j][k]) {
								if (min_coef_X[r] > coef_Y[j][k][r]) {
									min_coef_X[r] = coef_Y[j][k][r];
									temp_k = k; // select the station
									temp_j = j;
								}
							}
						}
					}
					//if (min_coef_X[r]>coef_Y[NUM_FAC-1][NUM_STA-1][r]) break;
					min_Kindex[r] = temp_k;
					min_Jindex[r] = temp_j;
				}
				min_Kindex[r] = NUM_STA - 1;
				min_Jindex[r] = NUM_FAC - 1;

				//get which level the emergency facility should go ? is it necessary?
				temp_sum = numeric_limits<float>::infinity();
				temp1 = 0;
				r_temp = 0;
				for (r = 0; r < NUM_LEVEL; r++)
				{
					if (temp_sum > temp1 + demand[i] * PENAL * beta[r]) {
						temp_sum = temp1 + demand[i] * PENAL * beta[r];
						r_temp = r;
					}
					//else {break;}
					if (r < NUM_LEVEL - 1) 
						temp1 += min_coef_X[r];
				}
				Z_D += temp_sum;

				for (r = 0; r < r_temp; r++)
				{
					Y[i][min_Jindex[r]][min_Kindex[r]][r] = 1;
				}
				Y[i][NUM_FAC - 1][NUM_STA - 1][r_temp] = 1;
			}
			else {//using hungarian algorithm
				for (r = 0; r < NUM_LEVEL - 1; r++) {
					for (j = 0; j < NUM_FAC; j++) {
						for (k = 0; k < NUM_STA; k++)if (link[j][k]) {
							Y[i][j][k][r] = 0;
							coef_Y[j][k][r] = numeric_limits<float>::infinity();
							if (X_fixed[j] && X_mask[j])
							{
								coef_Y[j][k][r] = demand[i] * distance[k][i][j] * (1 - prob[k])*beta[r];
							}
							//{coef_Y[j][k][r]=demand[i]*distance[k][i][j]*beta[r]; }
							if ((!X_fixed[j]) && (!X_mask[j]))
							{
								coef_Y[j][k][r] = demand[i] * distance[k][i][j] * (1 - prob[k])*beta[r] + mu[i][j][k];
							}
							//{coef_Y[j][k][r]=demand[i]*distance[k][i][j]*beta[r]+mu[i][j][k];}
						}
					}
				}
				coef_Y[NUM_FAC - 1][NUM_STA - 1][NUM_LEVEL - 1] = demand[i] * PENAL*beta[NUM_LEVEL - 1];
				temp_k = temp_j = -1;

				//get which level the emergency facility should go ? is it necessary?

				temp_sum = numeric_limits<float>::infinity();
				temp1 = 0;
				r_temp = 0;
				for (r = 0; r < NUM_LEVEL; r++)
				{
					if (temp_sum > temp1 + demand[i] * PENAL*beta[r]) {
						temp_sum = temp1 + demand[i] * PENAL*beta[r];
						r_temp = r;
						for (r1 = 0; r1 < r_temp; r1++)min_Jindex[r1] = mateRJ[r1], min_Kindex[r1] = mateRK[r1];//undefined yet
					}
					if (r < NUM_LEVEL - 1) Hungarian(temp1, r + 1, link, mateRJ, mateRK, coef_Y);//undefined yet
				}
				Z_D += temp_sum;
				for (r = 0; r < r_temp; r++)
				{
					//XX_coef[j]+=min_coef_X[r]-p[i][j][min_kr_index[r]][r]*e[i][j][min_kr_index[r]];
					Y[i][min_Jindex[r]][min_Kindex[r]][r] = 1;//update Y
				}

				Y[i][NUM_FAC - 1][NUM_STA - 1][r_temp] = 1;

			}
		}

		// upper bound
		GetCost(distance, JAss, KAss, X_temp, Z_U);
		/*IloEnv env_BB;
		IloIntArray X_eval(env_BB,NUM_FAC);
		for (j = 0; j < NUM_FAC; j++) {
			X_eval[j] = X_temp[j];
		}
		Eval_Problem(distance,link,X_eval,1,Z_U);*/
		//cout << Z_U << "\t" << Z_D << "\t" << Z_U_All << "\t" << Z_D_All << endl;

		// update the step size lamb
		tempxx = abs(Z_D - Z_D_All) / Z_D_All; // check if the lowerbound is improving
		if (tempxx < 0.001f)
			convergenceCount++;
		else
			convergenceCount = 0;
		if (Iteration > 200) {
			if (worst_Z_D > Z_D)
				worst_Z_D = Z_D;
			if (best_Z_D < Z_D)
				best_Z_D = Z_D;
			Count_1++;
			if (Count_1 >= 20) {
				if ((best_Z_D - worst_Z_D) / best_Z_D > 0.02f)
					Lamb = Lamb / dec_ratio;
				if ((best_Z_D - worst_Z_D) / best_Z_D < 0.001f)
					Lamb = Lamb * ((dec_ratio - 1) / 2 + 1);
				worst_Z_D = numeric_limits<float>::infinity();
				best_Z_D = -numeric_limits<float>::infinity();
				Count_1 = 0;
			}
		}
		else {
			if (Z_D > Z_D_All) {
				Count_1 = 0;// if updating step size lamb is necessary
				Count++; // if lowerbound is improving count++
			}
			else {
				Count_1++;
				Count = 0;
			}
			if (Count_1 > 5) {
				Lamb = Lamb / 1.05f;
				Count_1 = 0;
			}
			if (Count > 5) {
				Lamb = Lamb * 1.03f;
				Count = 0;
			}
		}

		// update best bounds
		if (Z_U < Z_U_All) {
			Z_U_All = Z_U;
			for (j = 0; j < NUM_FAC; j++) {
				X[j] = X_temp[j];
			}
		}
		if (Z_D > Z_D_All) {
			Z_D_All = Z_D;
		}
		if ((Z_U_All - Z_D_All) / Z_U_All < 0.001) {
			break;
		}

		// update multipliers
		float step_sum = 0.00001, step1_sum;
		for (i = 0; i < NUM_CUS; i++) {
			for (j = 0; j < NUM_FAC - 1; j++) {
				for (k = 0; k < NUM_STA - 1; k++) {
					if (link[j][k] && (!X_mask[j])) {
						step1_sum = 0;
						for (r = 0; r < NUM_LEVEL - 1; r++) {
							step1_sum += (float)Y[i][j][k][r];
						}
						step_sum += (step1_sum - (float)X_temp[j])*(step1_sum - (float)X_temp[j]);
					}
				}
			}
		}
		tk = Lamb * (Z_U_All - Z_D_All) / step_sum;
		for (i = 0; i < NUM_CUS; i++) {	//determine next lambda
			for (j = 0; j < NUM_FAC - 1; j++) {
				for (k = 0; k < NUM_STA - 1; k++) {
					if (link[j][k] && (!X_mask[j])) {
						step1_sum = 0;
						for (r = 0; r < NUM_LEVEL - 1; r++) {
							step1_sum += Y[i][j][k][r];
						}
						mu[i][j][k] -= tk*(-step1_sum + X_temp[j]);
						mu[i][j][k] = mu[i][j][k]>0 ? mu[i][j][k] : 0;
					}
					else {
						mu[i][j][k] = 0;
					}
				}
			}
		}
	}
	Z_D_new = Z_D_All;
	Z_U_new = Z_U_All;
	GetCost(distance, JAss, KAss, X_temp, Z_U);

	delete[] min_coef_X;
	delete[] min_Jindex;
	delete[] min_Kindex;
	delete[] X_temp;
	delete[] mateRJ;
	delete[] mateRK;
	for (i = 0; i<NUM_CUS; i++) {
		for (j = 0; j<NUM_FAC; j++) {
			for (k = 0; k<NUM_STA; k++)delete[] Y[i][j][k];
			delete[] Y[i][j];
		}
		delete[] Y[i];
	}
	delete[] Y;
	for (j = 0; j<NUM_FAC; j++) {
		for (k = 0; k<NUM_STA; k++) delete[] coef_Y[j][k];
		delete[] coef_Y[j];
	}
	delete[] coef_Y;
}



// Function to obtain feasible solution
void GetCost(const vector<vector<vector<float>>> &distance, const vector<vector<int>> &JAss, const vector<vector<int>> &KAss, bool* X, float &Z_U) {
	//C_f fixed cost, C_t transportation cost 
	Z_U = 0;
	int i, j, k, r, flag;
	float temp;
	for (j = 0; j < NUM_FAC - 1; j++) {
		if (X[j]) {
			Z_U += fixed_cost[j];
		}
	}
	int *min_Kindex = new int[NUM_LEVEL];
	for (r = 0; r < NUM_LEVEL; r++)
		min_Kindex[r] = -1;

	for (i = 0; i < NUM_CUS; i++) {
		for (r = 0; r < NUM_LEVEL; r++)
			min_Kindex[r] = -1;
		r = 0;
		temp = 1;
		for (int Jiter = 0; Jiter < JAss[i].size(); Jiter++) {
			j = JAss[i][Jiter];
			k = KAss[i][Jiter];
			flag = 0;
			for (int r1 = 0; r1 < r; r1++) {
				if (k == min_Kindex[r1]) {
					flag = 1;
					break;
				} //a station can only serve one level
			}
			if (flag) continue;

			if (X[j]) {
				Z_U += (1 - prob[k]) * temp * demand[i] * distance[k][i][j];
				//cout << j << k << endl;
				min_Kindex[r] = k;
				r++;
				temp = temp * prob[k];
				if (r >= NUM_LEVEL - 1)
					break;
			}
		}
		if (k != NUM_STA - 1) {
			Z_U += temp*demand[i] * PENAL;
		}
	}
	delete[] min_Kindex;
}



int Eval_Problem(const vector<vector<vector<float>>> & distance, const vector<vector<int>> &link,
	const IloIntArray x_val, const IloInt print, float &totalcost) {

	IloEnv env;
	totalcost = 0;
	std::ofstream output;
	output.open("Eval_" + to_string(static_cast<long long>(NUM_CUS)) + "x" + to_string(static_cast<long long>(NUM_FAC)) + ".txt", std::ios::app);
	//ofstream output("Model_Detailed_0708.txt");
	IloNumD4 y_val(env, NUM_CUS);
	IloNumD4 p_val(env, NUM_CUS);
	IloNumD4 w_val(env, NUM_CUS);

	for (int i = 0; i < NUM_CUS; i++) {
		y_val[i] = IloNumD3(env, NUM_FAC);
		p_val[i] = IloNumD3(env, NUM_FAC);
		w_val[i] = IloNumD3(env, NUM_FAC);
		for (int j = 0; j < NUM_FAC; j++) {
			y_val[i][j] = IloNumD2(env, NUM_STA);
			p_val[i][j] = IloNumD2(env, NUM_STA);
			w_val[i][j] = IloNumD2(env, NUM_STA);
			for (int k = 0; k < NUM_STA; k++) {
				y_val[i][j][k] = IloNumArray(env, NUM_LEVEL);
				p_val[i][j][k] = IloNumArray(env, NUM_LEVEL);
				w_val[i][j][k] = IloNumArray(env, NUM_LEVEL);
			}
		}
	}
	try {
		int i = 0;
		IloEnv env1;

		//Decision Variables
		//Binary Variables				
		//IloNumVarArray x(env1);
		IloNumVarArray3 y(env1);

		//Continuous Variables
		IloNumVarArray3 p(env1);
		IloNumVarArray3 w(env1);
		//IloNumVar cost(env1,0, IloInfinity, ILOfloat, "cost"); // common flow multiplier

		y.clear();
		y = IloNumVarArray3(env1, NUM_FAC);
		for (int j = 0; j < NUM_FAC; j++)
		{
			y[j] = IloNumVarArray2(env1, NUM_STA);
			for (int k = 0; k < NUM_STA; k++)
			{
				y[j][k] = IloNumVarArray(env1);
				for (int l = 0; l < NUM_LEVEL; l++)
				{
					string name = "y_";
					stringstream ss2;
					ss2 << j;
					name += ss2.str();
					stringstream ss3;
					ss3 << k;
					name += ss3.str();
					stringstream ss4;
					ss4 << l;
					name += ss4.str();
					y[j][k].add(IloNumVar(env1, 0, 1, ILOINT, name.c_str()));
				}
			}
		}


		p.clear();

		p = IloNumVarArray3(env1, NUM_FAC);
		for (int j = 0; j < NUM_FAC; j++)
		{
			p[j] = IloNumVarArray2(env1, NUM_STA);
			for (int k = 0; k < NUM_STA; k++)
			{
				p[j][k] = IloNumVarArray(env1);
				for (int l = 0; l < NUM_LEVEL; l++)
				{
					string name = "p_";
					stringstream ss2;
					ss2 << j;
					name += ss2.str();
					stringstream ss3;
					ss3 << k;
					name += ss3.str();
					stringstream ss4;
					ss4 << l;
					name += ss4.str();
					p[j][k].add(IloNumVar(env1, -IloInfinity, IloInfinity, ILOFLOAT, name.c_str()));
				}
			}
		}


		w.clear();

		w = IloNumVarArray3(env1, NUM_FAC);
		for (int j = 0; j < NUM_FAC; j++)
		{
			w[j] = IloNumVarArray2(env1, NUM_STA);
			for (int k = 0; k < NUM_STA; k++)
			{
				w[j][k] = IloNumVarArray(env1);
				for (int l = 0; l < NUM_LEVEL; l++)
				{
					string name = "w_";
					stringstream ss2;
					ss2 << j;
					name += ss2.str();
					stringstream ss3;
					ss3 << k;
					name += ss3.str();
					stringstream ss4;
					ss4 << l;
					name += ss4.str();
					w[j][k].add(IloNumVar(env1, -IloInfinity, IloInfinity, ILOFLOAT, name.c_str()));
				}
			}
		}

		IloModel model(env1);
		IloExpr expr(env1);

		// Constraints
		// Constraint sets 1

		for (int j = 0; j < NUM_FAC - 1; j++)
		{
			for (int k = 0; k < NUM_STA - 1; k++)
			{
				if (link[j][k]) {

					IloExpr Expr1(env1);
					//model.add(IloSum(y[i][j][k])-x_val[j]<=0);
					for (int r = 0; r < NUM_LEVEL - 1; r++)
					{
						Expr1 += y[j][k][r];
					}
					model.add(Expr1 - x_val[j] <= 0);
					Expr1.end();
				}
			}
		}

		// Constraint sets 2

		/*for(int j=0;j<NUM_FAC;j++)
		{
			for(int k=0;k<NUM_STA;k++)
			{
				for(int r=0;r<NUM_LEVEL;r++)
				{
					if (link[j][k]==0){
						y[j][k][r].setBounds(0,0);
						w[j][k][r].setBounds(0,0);
					}
					//model.add(y[j][k][r]-link[j][k]<=0);
				}
			}
		}*/
		//cout<<"Constraint sets 2 added"<<endl;
		//Constraint sets 3

		for (int k = 0; k < NUM_STA - 1; k++)
		{
			IloExpr expr1(env1);
			for (int j = 0; j < NUM_FAC - 1; j++)
			{
				for (int r = 0; r < NUM_LEVEL - 1; r++)
				{
					if (link[j][k]) {
						expr1 += y[j][k][r];
					}
				}
			}
			model.add(expr1 <= 1);
			expr1.end();
		}

		//cout<<"Constraint sets 3 added"<<endl;

		//Constraint sets 4

		IloExpr expr1(env1);
		for (int r = 0; r < NUM_LEVEL; r++)
		{
			expr1 += y[NUM_FAC - 1][NUM_STA - 1][r];
		}
		model.add(expr1 == 1);
		expr1.end();

		//cout<<"Constraint sets 4 added"<<endl;

		//Constraint sets 5

		for (int r = 0; r < NUM_LEVEL; r++)
		{
			IloExpr expr1(env1);
			for (int j = 0; j < NUM_FAC - 1; j++)
			{
				for (int k = 0; k < NUM_STA - 1; k++)
				{
					if (link[j][k]) { 
						expr1 += y[j][k][r]; 
					}
				}
			}
			for (int r1 = 0; r1 <= r; r1++)
			{
				expr1 += y[NUM_FAC - 1][NUM_STA - 1][r1];
			}
			model.add(expr1 == 1);
			expr1.end();
		}

		//cout<<"Constraint sets 5 added"<<endl;

		//Constraint sets 6

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					IloExpr expr1(env1);
					expr1 += (1 - prob[k]);
					model.add(expr1 == p[j][k][0]);//p is Z in paper
					expr1.end();
				}
			}
		}

		//cout<<"Constraint sets 6 added"<<endl;

		//Constraint sets 7

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 1; r < NUM_LEVEL; r++)
					{
						IloExpr expr1(env1);
						for (int j1 = 0; j1 < NUM_FAC - 1; j1++)
						{
							for (int k1 = 0; k1 < NUM_STA - 1; k1++)
							{
								if (link[j1][k1]) { expr1 += w[j1][k1][r - 1] * prob[k1] / (1 - prob[k1]); }
							}
						}
						expr1 *= (1 - prob[k]);
						model.add(expr1 == p[j][k][r]);
						expr1.end();
					}
				}
			}
		}
		//cout<<"Constraint sets 7 added"<<endl;

		// Constraint sets 8
		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 0; r < NUM_LEVEL; r++)
					{
						IloExpr expr1(env1);
						expr1 = w[j][k][r];
						model.add(expr1 <= p[j][k][r] + 2 * (1 - y[j][k][r]));
						expr1.end();
					}
				}
			}
		}
		//cout<<"Constraint sets 8 added"<<endl;

		// Constraint sets 9

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 0; r < NUM_LEVEL; r++)
					{
						IloExpr expr1(env1);
						expr1 = w[j][k][r];
						model.add(expr1 <= y[j][k][r]);
						expr1.end();
					}
				}
			}
		}

		//cout<<"Constraint sets 9 added"<<endl;

		// Constraint sets 10

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 0; r < NUM_LEVEL; r++)
					{
						IloExpr expr1(env1);
						expr1 = w[j][k][r];
						model.add(expr1 >= -1 * y[j][k][r]);
						expr1.end();
					}
				}
			}
		}

		//cout<<"Constraint sets 10 added"<<endl;

		// Constraint sets 11

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 0; r < NUM_LEVEL; r++)
					{
						IloExpr expr1(env1);
						expr1 = w[j][k][r];
						model.add(expr1 >= p[j][k][r] + 2 * (y[j][k][r] - 1));
						expr1.end();
					}
				}
			}
		}

		//cout<<"Constraint sets 11 added"<<endl;

		//Objective				

		for (int j = 0; j < NUM_FAC; j++)
		{
			for (int k = 0; k < NUM_STA; k++)
			{
				if (link[j][k]) {
					for (int r = 0; r < NUM_LEVEL; r++)
					{
						expr += demand[i] * distance[k][i][j] * w[j][k][r];

					}
				}
			}
		}

		// Objective
		model.add(IloMinimize(env1, expr));
		//cout<<"Objective added"<<endl;
		expr.end();
		//model.add(cost==expr);
		//cout<<"Objective function added"<<endl;

		IloCplex cplex(model);

		//env1.out() << cplex.getNbinVars() <<"\t" <<cplex.getNcols() << "\t" <<cplex.getNrows()<<endl;

		//cplex.exportModel("model.lp");

		//cplex.extract(model);
		cplex.setParam(IloCplex::Threads, 4);
		cplex.setParam(IloCplex::WorkMem, 1024);
		cplex.setParam(IloCplex::AdvInd, 0);
		//cplex.setParam(IloCplex:: NodeFileInd ,3);
		//cplex.setParam(IloCplex:: TreLim, 20000);
		cplex.setParam(IloCplex::TiLim, 300);
		//cplex.setParam(IloCplex:: EpAGap, 0.01);
		cplex.setParam(IloCplex::EpGap, 0.0001);
		cplex.setOut(env1.getNullStream());
		cplex.setWarning(env1.getNullStream());
		cplex.setError(env1.getNullStream());

		if (!cplex.solve()) {
			env1.error() << "Failed to optimize LP." << endl;
			throw(-1);
		}

		totalcost += cplex.getObjValue();

		for (int j = 0; j < NUM_FAC; j++) {
			for (int k = 0; k < NUM_STA; k++) {
				for (int r = 0; r < NUM_LEVEL; r++) {
					if (link[j][k]) {
						y_val[i][j][k][r] = cplex.getValue(y[j][k][r]);
						p_val[i][j][k][r] = cplex.getValue(p[j][k][r]);
						w_val[i][j][k][r] = cplex.getValue(w[j][k][r]);
					}
					else {
						y_val[i][j][k][r] = 0;
						p_val[i][j][k][r] = 0;
						w_val[i][j][k][r] = 0;
					}
				}
			}
		}

		for (int i = 1; i < NUM_CUS; i++)
		{
			//Objective
			IloExpr expr(env1);

			for (int j = 0; j < NUM_FAC; j++)
			{
				for (int k = 0; k < NUM_STA; k++)
				{
					if (link[j][k]) {
						for (int r = 0; r < NUM_LEVEL; r++)
						{
							expr += demand[i] * distance[k][i][j] * w[j][k][r];
						}
					}
				}
			}
			cplex.getObjective().setExpr(expr);
			cplex.solve();

			totalcost += cplex.getObjValue();

			for (int j = 0; j < NUM_FAC; j++) {
				for (int k = 0; k < NUM_STA; k++) {
					for (int r = 0; r < NUM_LEVEL; r++) {
						if (link[j][k]) {
							y_val[i][j][k][r] = cplex.getValue(y[j][k][r]);
							p_val[i][j][k][r] = cplex.getValue(p[j][k][r]);
							w_val[i][j][k][r] = cplex.getValue(w[j][k][r]);
						}
						else {
							y_val[i][j][k][r] = 0;
							p_val[i][j][k][r] = 0;
							w_val[i][j][k][r] = 0;

						}

					}
				}
			}
			// Objective
		}
		cplex.clearModel();
		model.end();
		cplex.end();
		env1.end();
		for (int j = 0; j < NUM_FAC; j++)
		{
			totalcost += fixed_cost[j] * x_val[j];
		}
		if (print) {
			output << endl << "totalcost=" << totalcost << endl;
			for (int i = 0; i < NUM_CUS; i++)
			{
				for (int j = 0; j < NUM_FAC; j++)
				{
					for (int k = 0; k < NUM_STA; k++)
					{
						for (int r = 0; r<NUM_LEVEL; r++)
						{
							if (y_val[i][j][k][r]>0.1)
							{
								output << "y" << i << j << k << r << "\t" << y_val[i][j][k][r] << "\t" << endl;
							}
						}
					}
				}
			}

			for (int i = 0; i < NUM_CUS; i++)
			{
				for (int j = 0; j < NUM_FAC; j++)
				{
					for (int k = 0; k < NUM_STA; k++)
					{
						for (int r = 0; r<NUM_LEVEL; r++)
						{
							if (y_val[i][j][k][r]>0.1)
							{
								output << "p" << i << j << k << r << "\t" << p_val[i][j][k][r] << "\t" << endl;
							}
						}
					}
				}
			}

			for (int i = 0; i < NUM_CUS; i++)
			{
				for (int j = 0; j < NUM_FAC; j++)
				{
					for (int k = 0; k < NUM_STA; k++)
					{
						for (int r = 0; r<NUM_LEVEL; r++)
						{
							if (y_val[i][j][k][r]>0.1)
							{
								output << "w" << i << j << k << r << "\t" << w_val[i][j][k][r] << "\t" << endl;
							}
						}
					}
				}
			}

			output << endl;
			output << endl;
		}
		output.close();
	}

	catch (IloException& e) {
		//output2 << "Error" << upper << "\t" << gap <<endl;
		cerr << "Concert exception caught: " << e << endl;
	}
	catch (...) {
		//output2 << "Error" << endl;
		cerr << "Unknown exception caught" << endl;
	}

	env.end();
	//system("pause");
	return 0;
}


void OneStepGreedy(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int> > &JAss,const vector<vector<int> > &KAss,
				   int &temp_j, float &cost_reduction,  bool* X, bool* X_mask, bool isInX){
	int j;
	
	float temp_cost_reduction,temp_cost;	
	float cost_original;
	bool *X_cur=new bool[NUM_FAC];
	X_cur[NUM_FAC-1]=1;
	
	for(j=0;j<NUM_FAC-1;j++)X_cur[j]=X[j]&X_mask[j];
	cost_reduction=-numeric_limits<float>::infinity();
	temp_j=-2;	
	GetCost(distance,JAss,KAss,X_cur, cost_original);
	/*IloEnv env_BB;
	IloIntArray X_eval(env_BB,NUM_FAC);
	for (j=0;j<NUM_FAC;j++) X_eval[j]=X_cur[j];
	Eval_Problem(distance,link,X_eval,0,cost_original);*/
	//Eval_P(distance,link,X_cur,0,cost_original);

	for(j=0;j<NUM_FAC-1;j++) if (!(X_mask[j])){
		
		if (isInX&&!X[j]) continue;
		X_cur[j]=1;
		GetCost(distance,JAss,KAss,X_cur, temp_cost);
		/*for (j=0;j<NUM_FAC;j++) X_eval[j]=X_cur[j];
		Eval_Problem(distance,link,X_eval,0,temp_cost);*/
//Eval_P(distance,link,X_cur,0,temp_cost);
		X_cur[j]=0;		
		temp_cost_reduction=cost_original-temp_cost;
		if(cost_reduction<temp_cost_reduction){
			cost_reduction=temp_cost_reduction;
			temp_j=j;
		}		
	}//for(j=0;j<J;j++) if (!(x_mask[j]))
	// the following comparasion was not in FEUL
	/*if ((cost_original-cost_reduction<Z_U_Global)&& (temp_j!=-1)){
		Z_U_Global=cost_original-cost_reduction;
		for(j=0;j<NUM_FAC-1;j++)X[j]=X[j]&X_mask[j];
		

		X[temp_j]=1;
for(j=0;j<NUM_FAC-1;j++)x_opt[j]=X[j];
	}*/
	delete [] X_cur;
}


void BB(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,const vector<vector<int> > &JAss,const vector<vector<int> > &KAss,
		bool* X, float &Z_D_new, float &Z_U_new, float ***mu,  bool* X_fixed, bool* X_mask , int j_mask){
			
	int i, j,k, tempCount, this_Node;
    int isReturn=0; 
	int reason=0;
	if (j_mask>=0) X_mask[j_mask]=1;
	else if (j_mask==-2)isReturn=1;
    Count_Tree++;
	this_Node=Count_Tree;
	//if (j_mask>0){cout<<"this node="<<this_Node<<" x_fixed["<<j_mask<<"]="<<X_fixed[j_mask]<<endl;}
    tempCount=0;
	for (j=0;j<NUM_FAC-1;j++)tempCount+=X_mask[j];
	
	if (tempCount==NUM_FAC-1){		
        for (j=0;j<NUM_FAC-1;j++) X[j]=X_fixed[j]&X_mask[j];

		GetCost(distance,JAss, KAss,X,Z_U_new);
		//Eval_P(distance,link,X,0,Z_D_new);
		Z_U_new=Z_D_new;
		isReturn=1;
		//cout<<"All installations are masked"<<endl;
		reason=1;
	}

	/*tempCount=0;
	for (j=0;j<NUM_FAC-1;j++)tempCount+=X_fixed[j]*X_mask[j];
	if (tempCount==NUM_OPT){		
        for (j=0;j<NUM_FAC-1;j++) X[j]=X_fixed[j]*X_mask[j];
		GetCost(distance,JAss, KAss,X,Z_D_new);
		Z_U_new=Z_D_new;
		isReturn=1;
		cout<<"All installations are masked"<<endl;
		reason=1.1;
	}
	tempCount=0;
	for (j=0;j<NUM_FAC-1;j++)tempCount+=(1-X_fixed[j])*X_mask[j];
	if (tempCount==NUM_FAC-1-NUM_OPT){		
        for (j=0;j<NUM_FAC-1;j++) X[j]=(1-X_fixed[j])*X_mask[j];
		X[NUM_FAC-1]=1;
		GetCost(distance,JAss, KAss,X,Z_D_new);
		Z_U_new=Z_D_new;
		isReturn=1;
		cout<<"All installations are masked"<<endl;
		reason=1.2;
	}*/

	
	if(!isReturn) {

		//cout<<" beforeLR ZU: "<<cost_UB_new<<"; ZD= "<<cost_LB_new<<endl;

		
		SolveLR(distance,link,JAss,KAss,X,Z_D_new, Z_U_new, mu, X_fixed, X_mask,Num_Iteration,1.5,1.05);
		//SolveLR(X,Z_D_new, Z_U_new, mu, X_fixed, X_mask,200,.5);
		//cout<<"this node="<<this_Node<<" updated ZU: "<<Z_U_new<<"; ZD= "<<Z_D_new<<endl;
		//cout<<" updated ZU: "<<Z_U_new<<"; ZD= "<<Z_D_new<<endl;
		//system("pause");
#if _DEBUG
		//getchar();
		if (this_Node==8){
			cout <<"Special Node: "<<this_Node<<endl;
			cout<<" updated ZU: "<<cost_UB_new<<"; ZD= "<<cost_LB_new<<endl;
			cout << "facilities : " ;
			for(j=0;j<J;j++)if(X[j])cout<<j<<" ";
			cout<<endl;
		}
#endif
	}

	if (Z_U_Global>Z_U_new)	{
		Z_U_Global=Z_U_new;
		for(j=0;j<NUM_FAC-1;j++)x_opt[j]=X[j];
	}

	if (Count_Tree>Count_Tree_Lim) {//cout<<"!Node number reaches its minumum, BB is ending!\n";
        isReturn=1;      
		cout<<"Count_Tree is too large!"<<endl;
		reason=3;
    }
    endTime=(float)clock();
    if((endTime-beginTime)/CLOCKS_PER_SEC>timeLimit){
		//cout<<endTime<<"\t"<<beginTime<<endl;
        //cout<<"!time is up, BB is ending!\n";
        isReturn=1; 
		reason=4;
    }
	//it has potential to branch
	if (!isReturn&&(Z_U_Global-Z_D_new)/Z_U_Global>-0.000){//&&Z_U_Global>Z_D_new
		//heuristically determine which variable to branch;
		int temp_j;
		float  cost_reduction;
		
		OneStepGreedy(distance,link,JAss,KAss,temp_j,cost_reduction, X, X_mask,0);

		float Z_D_new_L, Z_U_new_L, Z_D_new_R, Z_U_new_R;
		Z_D_new_L=Z_D_new_R=Z_D_new;
		//Z_U_new_L=Z_U_new_R=numeric_limits<float>::infinity();//orignal
		Z_U_new_L=Z_U_new_R=Z_U_new;

		bool * X_L= new bool [NUM_FAC];
		bool * X_R= new bool [NUM_FAC];
		for(j=0;j<NUM_FAC;j++)X_L[j]=X_R[j]=X[j];
		//X_L[J]=X_R[J]=1;	
		float *** mu_r=new float ** [NUM_CUS];  
		for(i=0;i<NUM_CUS;i++){
			mu_r[i]=new float *[NUM_FAC-1];  
			for(j=0;j<NUM_FAC-1;j++){
				mu_r[i][j]=new float [NUM_STA-1];
				for (k=0;k<NUM_STA-1;k++){
					mu_r[i][j][k]=mu[i][j][k];
				}
			}			
		}

		X_fixed[temp_j]=1; 
		//cout <<"Node: "<<this_Node<<" going LEFT!"<<endl;
		BB(distance,link,JAss,KAss,X_L,Z_D_new_L, Z_U_new_L, mu,X_fixed, X_mask, temp_j);
		
		//cout <<"Node: "<<this_Node<<" going RIGHT!"<<endl;	
		X_fixed[temp_j]=0;        
		BB(distance,link,JAss,KAss,X_R, Z_D_new_R, Z_U_new_R, mu_r, X_fixed, X_mask,temp_j);
		 
		int isUpdate=0;
		if (Z_U_new_L< Z_U_new){
			Z_U_new=Z_U_new_L;
			isUpdate=-1;
		}
		if (Z_U_new_R< Z_U_new){ 
			Z_U_new=Z_U_new_R;
			isUpdate=1;
		}
		if (isUpdate){
			if (isUpdate<0) for (j=0;j<NUM_FAC;j++)X[j]=X_L[j];
			else for (j=0;j<NUM_FAC;j++)X[j]=X_R[j];
		}		
		//if (Z_U_Global>Z_U_new)Z_U_Global=Z_U_new;

		
	/*	cout << "facilities after  update: " ;
		for(j=0;j<J;j++)if(X[j])cout<<j<<" ";
			cout<<endl;

		cout<<" updated ZU= "<<Z_U_new<<"; Z_U_L= "<<Z_U_new_L<<"; Z_U_R = "<<Z_U_new_R<<endl;
		
			getchar();*/

		Z_D_new_L=Z_D_new_L<Z_D_new_R?Z_D_new_L:Z_D_new_R;
		Z_D_new=Z_D_new>Z_D_new_L?Z_D_new:Z_D_new_L;      
		

		//delete variables

		for(i=0;i<NUM_CUS;i++){
			for(j=0;j<NUM_FAC-1;j++){
				delete [] mu_r[i][j];
			}
			delete [] mu_r[i];
		}
		delete [] mu_r;
		delete [] X_L;
		delete [] X_R;

	}
	else
	{
		if (!isReturn){
			//cout<<"Node: "<<this_Node<<",this branch is trimmed!"<<endl;
			//cout<<"reason: "<<reason<<endl;
		}
	}
	//if (Count_Tree==1)Z_U_G=cost_UB_new;
		
	//cout<<"Node: "<<this_Node<<"; Z_D = "<<Z_D_new<<"; Feaible solution = "<<Z_U_new<<endl;
//	cout << "facilities: " ;
			//for(j=0;j<J;j++)if(X[j])cout<<j<<" ";
			//cout<<endl;
	if (j_mask>=0) X_mask[j_mask]=0;

}



void BBBFS(const vector<vector<vector<float>>> &distance,const vector<vector<int>> &link,const vector<vector<int>> &JAss,const vector<vector<int>> &KAss,
		bool* X, float ***mu, BBNode &thisBBNode){

	int i, j, k, l, tempCount, reason, this_Node, num_path; // isTrim, trimCount;int j, isTerminate;this_Node; isReturn=0;, Level,,isUpdate

	while (1) {
		//reaches resourse limit 
		Count_Tree++;
		this_Node = Count_Tree;
		reason = 0;
		if (Count_Tree > Count_Tree_Lim) {
			cout << "!Node number reaches its minumum, BB is ending!\n";
			cout << "Count_Tree is too large!" << endl;
			reason = 2;
			break;
		}

		endTime = (float)clock();
		if ((endTime - beginTime) / CLOCKS_PER_SEC > timeLimit) {
			cout << endTime << "\t" << beginTime << endl;
			cout << "!time is up, BB is ending!\n";
			reason = 3;
			break;
		}

		tempCount = 0;
		for (j = 0; j < NUM_FAC - 1; j++)
			tempCount += thisBBNode.X_mask[j];
		if (tempCount == NUM_FAC - 1) {
			cout << "tempt_count=" << tempCount << endl; getchar();
			for (j = 0; j < NUM_FAC - 1; j++) 
				X[j] = thisBBNode.X_fixed[j] & thisBBNode.X_mask[j];

			GetCost(distance, JAss, KAss, X, thisBBNode.Z_U);
			/*IloEnv env_BB;
			IloIntArray X_eval(env_BB,NUM_FAC);
			for (j=0;j<NUM_FAC;j++) X_eval[j]=X[j];
			Eval_Problem(distance,link,X_eval,0,thisBBNode.Z_U);*/
			//thisBBNode.Z_D = thisBBNode.Z_U;
			//cout<<"All installations are masked"<<endl;
			reason = 1;
			break;
		}

		if (true) { //could run LR
			for (i = 0; i < NUM_CUS; i++) {		//initialize 
				for (j = 0; j < NUM_FAC - 1; j++) {
					l = 0;
					for (k = 0; k < NUM_STA - 1; k++)
						if (link[j][k]) {
						mu[i][j][k] = thisBBNode.muS[i][j][l];
						l++;
					}
				}
			}
			SolveLR(distance, link, JAss, KAss, X, thisBBNode.Z_D, thisBBNode.Z_U, mu, thisBBNode.X_fixed, thisBBNode.X_mask, Num_Iteration, 1.5, 1.05);
		}

		if (Z_U_Global > thisBBNode.Z_U) {
			Z_U_Global = thisBBNode.Z_U;
			for (j = 0; j < NUM_FAC - 1; j++) {
				x_opt[j] = X[j];
			}
		}
		if ((Z_U_Global - thisBBNode.Z_D) / (Z_U_Global + 0.00001)>-0.00001) { // Z_U_Global-thisBBNode.Z_D>0
		// heuristically determine which variable to branch;
			float  cost_reduction;

			OneStepGreedy(distance, link, JAss, KAss, thisBBNode.temp_j, cost_reduction, X, thisBBNode.X_mask, 0);

			BBNode thisBBNode_L, thisBBNode_R;

			thisBBNode_L.X_fixed = new bool[NUM_FAC];
			thisBBNode_L.X_mask = new bool[NUM_FAC];
			thisBBNode_R.X_fixed = new bool[NUM_FAC];
			thisBBNode_R.X_mask = new bool[NUM_FAC];
			thisBBNode_L.muS = new float **[NUM_CUS];//define lagrangian multiplier
			thisBBNode_R.muS = new float **[NUM_CUS];

			for (i = 0; i < NUM_CUS; i++) {	//initialize 
				thisBBNode_L.muS[i] = new float *[NUM_FAC - 1];
				thisBBNode_R.muS[i] = new float *[NUM_FAC - 1];
				for (j = 0; j < NUM_FAC - 1; j++) {
					num_path = 0;
					for (k = 0; k < NUM_STA - 1; k++)if (link[j][k])num_path++;
					thisBBNode_L.muS[i][j] = new float[num_path];
					thisBBNode_R.muS[i][j] = new float[num_path];
					l = 0;
					for (k = 0; k < NUM_STA - 1; k++) {
						if (link[j][k]) {
							thisBBNode_L.muS[i][j][l] = mu[i][j][k];
							thisBBNode_R.muS[i][j][l] = mu[i][j][k];
							l++;
						}
					}
				}
			}

			for (int j = 0; j < NUM_FAC; j++) {
				thisBBNode_L.X_fixed[j] = thisBBNode_R.X_fixed[j] = thisBBNode.X_fixed[j];
				thisBBNode_L.X_mask[j] = thisBBNode_R.X_mask[j] = thisBBNode.X_mask[j];
			}
			int temp_j = thisBBNode.temp_j;
			thisBBNode_L.X_mask[temp_j] = thisBBNode_R.X_mask[temp_j] = 1;
			thisBBNode_L.X_fixed[temp_j] = 1; thisBBNode_R.X_fixed[temp_j] = 0;
			thisBBNode_L.Z_D = thisBBNode_R.Z_D = thisBBNode.Z_D;
			thisBBNode_L.Z_U = thisBBNode_R.Z_U = thisBBNode.Z_U;
			thisBBNode_L.temp_j = thisBBNode_R.temp_j = -1;
			Live_queue.push(thisBBNode_L);
			Live_queue.push(thisBBNode_R);
		}
		else {
			//cout<<"Node: "<<this_Node<<",this branch is trimmed!"<<endl;
		}

		break;
	}
	if (reason >= 1) {
		Z_D_Global = Z_D_Global < thisBBNode.Z_D ? Z_D_Global : thisBBNode.Z_D;
	} // choose the lowest LB for the live nodes which haven't been branched
}



void Hungarian(float &result, int temp_R,const vector<vector<int>> &link,int *mateRJ, int *mateRK,float ***coef_Y){
	int i,j,k, r,s;
	int isBreak;
	int R_max=temp_R;
	float temp;
	list<int>::iterator childIter;
	//initialization
	float **Jmin=new float *[NUM_STA-1];
	for(k=0;k<NUM_STA-1;k++){
		Jmin[k]= new float[R_max];
		for(r=0;r<R_max;r++){
			Jmin[k][r]=NUM_FAC-1;
		} 
	}
	for(r=0;r<R_max;r++){
		for(k=0;k<NUM_STA-1;k++){
			temp=numeric_limits<float>::infinity();
			for(j=0;j<NUM_FAC-1;j++)if (link[j][k]){
				if (temp>coef_Y[j][k][r]){
					temp=coef_Y[j][k][r];
					Jmin[k][r]=j;
				}
			}	
		}
	}

	float *alphaH=new float [NUM_STA-1];
	float *betaH=new float [R_max];
	int *mateK=new int [NUM_STA-1];
	int *exposed=new int [NUM_STA-1];
	int *label=new int [NUM_STA-1];
	float *slack=new float [R_max];
	int *nhbor=new int [R_max];
	ANode * ATree=new ANode [NUM_STA-1];
	
	list<int> Q;
	
	for(k=0;k<NUM_STA-1;k++){
		alphaH[k]=0;
		mateK[k]=-1;
		ATree[k].parent=-1;
	}
	for(r=0;r<R_max;r++){
		mateRJ[r]=-1;
		mateRK[r]=-1;
		betaH[r]=numeric_limits<float>::infinity();
		for(k=0;k<NUM_STA-1;k++){
			j=Jmin[k][r];
			if(betaH[r]>coef_Y[j][k][r])betaH[r]=coef_Y[j][k][r];
		}
	}
	
	//solving the RP mathcing problem with augmenting path algorithm
	
		
	for(s=0;s<R_max;s++){
	//while(1){//R stages
		
		for(k=0;k<NUM_STA-1;k++){
			ATree[k].children.clear();
			ATree[k].parent=-1;
		}
		for(r=0;r<R_max;r++)	slack[r]=numeric_limits<float>::infinity();
		for(k=0;k<NUM_STA-1;k++)	{label[k]=-2;exposed[k]=-1;}
		for(k=0;k<NUM_STA-1;k++)for(r=0;r<R_max;r++){
			j=Jmin[k][r];
			if(abs(coef_Y[j][k][r]-alphaH[k]-betaH[r])<INFSMALL){
				if(mateRK[r]==-1) {if(exposed[k]<0)exposed[k]=r;}
				else if (mateRK[r]!=k){
				ATree[k].children.push_back(mateRK[r]);
				ATree[mateRK[r]].parent=k;
				}
			}
		}
		
		isBreak=0;
		Q.clear();
		for(k=0;k<NUM_STA-1;k++)if(mateK[k]==-1){			
			if(exposed[k]>=0) {
				Augment(k,exposed,label,mateRK,mateK);
				isBreak=1;
				break;
			}
			Q.push_back(k);
			label[k]=-1;
			for(r=0;r<R_max;r++) {
				j=Jmin[k][r];
				temp=coef_Y[j][k][r]-alphaH[k]-betaH[r];
				if(temp<slack[r] &&temp>0)slack[r]=temp,nhbor[r]=k;
			}
		}
		//index=0;
		while(1){
			//index++;

			
			if(!isBreak) while(Q.size()){
				i=Q.back();
				Q.pop_back();
				childIter=ATree[i].children.begin();
				while(childIter!=ATree[i].children.end()){
					k=*childIter;
					if(label[k]<=-1){
						label[k]=i;
						Q.push_back(k);
						if(exposed[k]>=0){
							Augment(k,exposed,label,mateRK,mateK);
							isBreak=1;
							break;
						}
						for(r=0;r<R_max;r++){
							j=Jmin[k][r];
							temp=coef_Y[j][k][r]-alphaH[k]-betaH[r];
							if(temp>0 && temp<slack[r] )slack[r]=temp,nhbor[r]=k;
						}
					}

					childIter++;
				}//while(childIter!=ATree[i].children.end())	
				if(isBreak) break;
				
			}//if(!isBreak) while(Q.size()){
			//if(!isBreak) while(Q.size()){}
			if(!isBreak)isBreak=Modify(R_max,slack,alphaH,betaH,Q,ATree,exposed,label,nhbor,mateRK,mateK);
			if (isBreak) break;
			//if (index>=100) break;
		}
		//isBreak=1;
		//if(R_max<=J){
			//for(r=0;r<R_max;r++)if(mateR[r]<0){isBreak=0;break;}
		//}
		//else
		//{for(j=0;j<J;j++) if(mateJ[j]<0){isBreak=0;break;}
		//if(isBreak)	break;
	
		

	}//R stages
	result=0;
	for(k=0;k<NUM_STA-1;k++)if(mateK[k]>=0){
		//if (alphaH[k]<0){cout<<k<<"!";getchar();}
		result+=alphaH[k];
	}
	for(r=0;r<R_max;r++){
		result+=betaH[r];
		k=mateRK[r];
		j=Jmin[k][r];
		mateRJ[r]=j;
	}
#if _DEBUG
	int aa=1;
#endif 
	for(k=0;k<NUM_STA-1;k++) {
		delete [] Jmin[k];
	}
	delete [] Jmin;

	for(k=0;k<NUM_STA-1;k++)ATree[k].children.clear();
	delete [] ATree;
	
	delete [] alphaH; 
	delete [] betaH;
	delete [] mateK;
	delete [] exposed;
	delete [] label;
	delete [] slack;
	delete [] nhbor;
	
	Q.clear();
}



void Augment(int k,int *exposed, int *label, int *mateRK,int *mateK)
{
	if (label[k]<0){
		mateK[k]=exposed[k];
		mateRK[exposed[k]]=k;
	}
	else if (label[k]>=0){
		exposed[label[k]]=mateK[k];
		mateK[k]=exposed[k];
		mateRK[exposed[k]]=k;
		Augment(label[k],exposed,label,mateRK,mateK);
	}
	
}



int Modify(const int R_max,float *slack,float *alphaH, float *betaH,list<int>& Q,ANode *ATree,int *exposed, int *label, int *nhbor,int *mateRK,int *mateK)
{
	int k,r;
	int isBreak=0;
	float theta_2;
	float theta=numeric_limits<float>::infinity();
	for(r=0;r<R_max;r++)if(slack[r]>0&&theta>slack[r])
		theta_2=theta=slack[r];
	
	theta*=(float)1/2;
	for(k=0;k<NUM_STA-1;k++){
		if(label[k]>=-1)alphaH[k]+=theta;
		else alphaH[k]-=theta;
	}
	for(r=0;r<R_max;r++){
		if(slack[r]==0)betaH[r]-=theta;
		else betaH[r]+=theta;
	}
	for(r=0;r<R_max;r++)if(slack[r]>0){
		slack[r]-=theta_2;
		if(slack[r]==0){
			if(mateRK[r]==-1){
				exposed[nhbor[r]]=r;Augment(nhbor[r],exposed,label,mateRK,mateK);
				isBreak=1;
				break;
			}
			else{
				
				//label[mateR[r]]=nhbor[r];
				Q.push_back(nhbor[r]);
				ATree[nhbor[r]].children.push_back(mateRK[r]);
				ATree[mateRK[r]].parent=nhbor[r];

			}
		}
	}
	return isBreak;

	
}


// Sub_Problem is not used in this problem
int Sub_Problem(const vector<vector<vector<float>>> & distance,const vector<vector<int>> &link,
			   float ***lambda, const bool* x_fixed, const bool* x_mask, bool ****y_val, float &subcost,float &UB_sub2){
			
			subcost=0;
			UB_sub2=0;
			try {
				
					IloEnv env1;
					int i=0;
					
					//Decision Variables
					//Binary Variables				
					//IloNumVarArray x(env1);
					IloNumVarArray3 y(env1);

					//Continuous Variables
					IloNumVarArray3 p(env1);
					IloNumVarArray3 w(env1);
					//IloNumVar cost(env1,0, IloInfinity, ILOdouble, "cost"); // common flow multiplier

					y.clear();	
					y=IloNumVarArray3(env1, NUM_FAC);
					for(int j=0;j<NUM_FAC;j++)
					{
						y[j]=IloNumVarArray2(env1, NUM_STA);
						for(int k=0;k<NUM_STA;k++)
						{
							y[j][k]=IloNumVarArray(env1,NUM_LEVEL,0,1,ILOINT);
						}
					}
				

					p.clear();
					
					p=IloNumVarArray3(env1, NUM_FAC);
					for(int j=0;j<NUM_FAC;j++)
					{
						p[j]=IloNumVarArray2(env1, NUM_STA);
						for(int k=0;k<NUM_STA;k++)
						{
							p[j][k]=IloNumVarArray(env1,NUM_LEVEL,-IloInfinity,IloInfinity);
						}
					}


					w.clear();
				
					w=IloNumVarArray3(env1, NUM_FAC);
					for(int j=0;j<NUM_FAC;j++)
					{
						w[j]=IloNumVarArray2(env1, NUM_STA);
						for(int k=0;k<NUM_STA;k++)
						{
							w[j][k]=IloNumVarArray(env1,NUM_LEVEL,-IloInfinity,IloInfinity);
						}
					}

					IloModel model(env1);
					IloExpr expr(env1);
					
					// Constraints

					for(int j=0;j<NUM_FAC-1;j++)
					{
						if (x_mask[j]){
							for(int k=0;k<NUM_STA-1;k++)
							{
								if (link[j][k]){

								IloExpr Expr1(env1);
								//model.add(IloSum(y[i][j][k])-x_val[j]<=0);
								for(int r=0;r<NUM_LEVEL-1;r++)
								{
									Expr1+=y[j][k][r];
								
								}
								model.add(Expr1-x_fixed[j]<=0);
								Expr1.end();
								}
							}
						}
					}
					
					// Constraint sets 2
								
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							for(int r=0;r<NUM_LEVEL;r++)
							{
								if (link[j][k]==0){
									y[j][k][r].setBounds(0,0);
									w[j][k][r].setBounds(0,0);
								}
								//model.add(y[j][k][r]-link[j][k]<=0);
							}
						}
					}
					//cout<<"Constraint sets 2 added"<<endl;
					//Constraint sets 3
								
					for(int k=0;k<NUM_STA-1;k++)
					{
						IloExpr expr1(env1);
						for(int j=0;j<NUM_FAC-1;j++)
						{
							for(int r=0;r<NUM_LEVEL-1;r++)
							{
								if (link[j][k]){expr1 += y[j][k][r];}
							}
							//model.add(y[i][j][k][NUM_LEVEL-1]==0);
						}
						model.add(expr1<=1);
						expr1.end();
					}
					
					//cout<<"Constraint sets 3 added"<<endl;
					
					//Constraint sets 4
						
					IloExpr expr1(env1);
					for(int r=0;r<NUM_LEVEL;r++)
					{
						expr1 += y[NUM_FAC-1][NUM_STA-1][r];
					}
					model.add(expr1==1);
					expr1.end();
			
					//cout<<"Constraint sets 4 added"<<endl;
					
					//Constraint sets 5
									
					for(int r=0;r<NUM_LEVEL;r++)
					{
						IloExpr expr1(env1);
						for(int j=0;j<NUM_FAC-1;j++)
						{
							for(int k=0;k<NUM_STA-1;k++)
							{
								if (link[j][k]){expr1 += y[j][k][r];}
							}
						}
						for(int r1=0;r1<=r;r1++)
						{
							expr1 += y[NUM_FAC-1][NUM_STA-1][r1];
						}
						model.add(expr1==1);
						expr1.end();
					}
					
					//cout<<"Constraint sets 5 added"<<endl;
					
					//Constraint sets 6
								
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							IloExpr expr1(env1);
							expr1 += (1-prob[k]);
							model.add(expr1==p[j][k][0]);//p is Z in paper
							expr1.end();
							}
						}
					}
					
					//cout<<"Constraint sets 6 added"<<endl;
					
					//Constraint sets 7
								
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							for(int r=1;r<NUM_LEVEL;r++)
							{
								IloExpr expr1(env1);
								for(int j1=0;j1<NUM_FAC-1;j1++)
								{
									for(int k1=0;k1<NUM_STA-1;k1++)
									{
										if (link[j1][k1]){expr1 += w[j1][k1][r-1] * prob[k1]/(1-prob[k1]);}
									}
								}
								expr1 *= (1-prob[k]);
								model.add(expr1==p[j][k][r]);
								expr1.end();
							}
							}
						}
					}
					//cout<<"Constraint sets 7 added"<<endl;
					
					// Constraint sets 8
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							for(int r=0;r<NUM_LEVEL;r++)
							{
								IloExpr expr1(env1);
								expr1 = w[j][k][r];
								model.add(expr1<=p[j][k][r]+2*(1-y[j][k][r]));
								expr1.end();
							}
							}
						}						
					}
					//cout<<"Constraint sets 8 added"<<endl;
					
					// Constraint sets 9
									
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							for(int r=0;r<NUM_LEVEL;r++)
							{
								IloExpr expr1(env1);
								expr1 = w[j][k][r];
								model.add(expr1<=y[j][k][r]);
								expr1.end();
							}
							}
						}						
					}
					
					//cout<<"Constraint sets 9 added"<<endl;
					
					// Constraint sets 10
						
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							for(int r=0;r<NUM_LEVEL;r++)
							{
								IloExpr expr1(env1);
								expr1 = w[j][k][r];
								model.add(expr1>=-1*y[j][k][r]);
								expr1.end();
							}
							}
						}						
					}
					
					//cout<<"Constraint sets 10 added"<<endl;
					
					// Constraint sets 11
								
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
							for(int r=0;r<NUM_LEVEL;r++)
							{
								IloExpr expr1(env1);
								expr1 = w[j][k][r];
								model.add(expr1>=p[j][k][r]+2*(y[j][k][r]-1));
								expr1.end();
							}
							}
						}						
					}
					
					//cout<<"Constraint sets 11 added"<<endl;
					
					//Objective				
							
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
								for(int r=0;r<NUM_LEVEL;r++)
								{
									expr += demand[i]*distance[k][i][j]*w[j][k][r];
								}
							}
						}						
					}
									
					for(int j=0;j<NUM_FAC-1;j++)
					{
						for(int k=0;k<NUM_STA-1;k++)
						{
							if (link[j][k] && (!x_mask[j])){
								for(int r=0;r<NUM_LEVEL-1;r++)
								{
									expr += lambda[i][j][k]*y[j][k][r];
								}
							}
						}						
					}
					// Objective
					
					model.add(IloMinimize(env1, expr));
					expr.end();
					
					//cout<<"Objective added"<<endl;
					
					//model.add(cost==expr);
					//cout<<"Objective function added"<<endl;
					
					IloCplex cplex(model);

					//env1.out() << cplex.getNbinVars() <<"\t" <<cplex.getNcols() << "\t" <<cplex.getNrows()<<endl;

					//cplex.exportModel("model.lp");

					//cplex.extract(model);
					cplex.setParam(IloCplex:: Threads, 2);
					cplex.setParam(IloCplex:: WorkMem, 1024);
					cplex.setParam(IloCplex:: NodeFileInd ,3);
					//cplex.setParam(IloCplex:: TreLim, 20000);
					cplex.setParam(IloCplex:: TiLim, 600);
					//cplex.setParam(IloCplex::AdvInd, 2); //can be 0, 1 or 2
					//cplex.setParam(IloCplex:: EpAGap, 0.01);
					cplex.setParam(IloCplex:: EpGap, 0.01);
					cplex.setOut(env1.getNullStream());
					cplex.setWarning(env1.getNullStream());
					cplex.setError(env1.getNullStream());

					if (!cplex.solve() ) {
						env1.error() << "Failed to optimize LP." << endl;
						throw(-1);
					}
					subcost +=cplex.getObjValue();
					for(int j=0;j<NUM_FAC;j++){
						for(int k=0;k<NUM_STA;k++){
							for(int r=0;r<NUM_LEVEL;r++){
								if (link[j][k]){
									y_val[i][j][k][r]=cplex.getValue(y[j][k][r]);
									UB_sub2 += demand[i]*distance[k][i][j]*cplex.getValue(w[j][k][r]);

								}
								else {
									y_val[i][j][k][r]=0;
								}
							}
						}						
					}

				cout<<"customer";	
				for(i=1;i<NUM_CUS;i++)
				{
					cout<<i<<"\t";
					
					//modify the Objective
					IloExpr expr(env1);
							
					for(int j=0;j<NUM_FAC;j++)
					{
						for(int k=0;k<NUM_STA;k++)
						{
							if (link[j][k]){
								for(int r=0;r<NUM_LEVEL;r++)
								{
									expr += demand[i]*distance[k][i][j]*w[j][k][r];
								}
							}
						}						
					}
									
					for(int j=0;j<NUM_FAC-1;j++)
					{
						for(int k=0;k<NUM_STA-1;k++)
						{
							if (link[j][k] && (!x_mask[j])){
								for(int r=0;r<NUM_LEVEL-1;r++)
								{
									expr += lambda[i][j][k]*y[j][k][r];
								}
							}
						}						
					}
					cplex.getObjective().setExpr(expr);
					// Objective
					cplex.solve();
					
					subcost +=cplex.getObjValue();
					for(int j=0;j<NUM_FAC;j++){
						for(int k=0;k<NUM_STA;k++){
							for(int r=0;r<NUM_LEVEL;r++){
								if (link[j][k]){
									y_val[i][j][k][r]=cplex.getValue(y[j][k][r]);
									UB_sub2 += demand[i]*distance[k][i][j]*cplex.getValue(w[j][k][r]);

								}
								else {
									y_val[i][j][k][r]=0;
								}
							}
						}						
					}

					
				}
				cout<<"end of sub"<<endl;
				cplex.clearModel();
				model.end();
				cplex.end();
				env1.end();
				
			}
			catch (IloException& e) {
				//output2 << "Error" << upper << "\t" << gap <<endl;
				cerr << "Concert exception caught: " << e << endl;
			}
			catch (...) {
				//output2 << "Error" << endl;
				cerr << "Unknown exception caught" << endl;
			}
		
			
			//}
			//output << "b= "<< count1 <<  endl;
			//}
			//output << "Solution status = " << status << endl;
			//output << "Solution value = " << min << endl;
			//output << "Solution count = " << count1 << endl;
			//}

			//}
			//}
			//system("pause");
			return 0;
}


void PrintCost(const vector<vector<vector<float>>> &distance,const vector<vector<int> > &JAss,const vector<vector<int> > &KAss, 
			 bool* X, float &Z_U, bool print){ //,float *C_f,float *C_t
	
	int i,j,k,r,flag;
	std::ofstream output;
	output.open("Print_"+to_string(static_cast<long long>(NUM_CUS))+"x"+to_string(static_cast<long long>(NUM_FAC))+".txt",std::ios::app);
	if (print)for(j=0;j<NUM_FAC;j++){output << "x"<<j<< "\t"<< X[j] << "\t" << endl;}
	Z_U=0;
	float temp ;
	for(j=0;j<NUM_FAC-1;j++)if(X[j])Z_U+=fixed_cost[j];
	int *min_Kindex =new int [NUM_LEVEL];
	for (r=0;r<NUM_LEVEL;r++)min_Kindex[r]=-1;
	
	for(i=0;i<NUM_CUS;i++){
		for (r=0;r<NUM_LEVEL;r++)min_Kindex[r]=-1;
		r=0;
		temp=1;

		for(int Jiter=0;Jiter<JAss[i].size();Jiter++){
			j=JAss[i][Jiter];
			k=KAss[i][Jiter];
			flag=0;
			for (int r1=0;r1<r;r1++){
				if (k==min_Kindex[r1]) {flag=1;break;} //a station can only serve one level
			}
			if (flag) continue;
		
			if(X[j]){		
				Z_U+=(1-prob[k])*temp*demand[i]*distance[k][i][j];
				min_Kindex[r]=k;
				if (print) output << "p_"<<i<<"_"<<j<<"_"<<k<<"_"<<r<<"\t"<< (1-prob[k])*temp << "\t" <<endl;
				//cout<<i<<" "<<j<<" "<<k<<" "<<r<<endl;
				//if(	C_f&&C_t)C_t[r]+=(1-prob[k])*temp*demand[i]*distance[k][i][j];
				r++;
				temp=temp*prob[k];
				if(r>=NUM_LEVEL-1)break;	
			}
			//Kiter++;
		}
		if(k!=NUM_STA-1) {
			Z_U+=temp*demand[i]*PENAL;
			if (print) output << "p_" << i << "_" << j << "_" << k << "_" << r << "\t" << (1 - prob[k])*temp << "\t" << endl;
			//cout<<i<<" "<<NUM_FAC-1<<" "<<NUM_STA-1<<" "<<r<<endl;
			//if(	C_f&&C_t)C_t[r]+=temp*demand[i]*PENAL;
		}
		//IJiter++;IKiter++;
		

	}
	delete [] min_Kindex;
	if (print){output <<endl<<"totalcost="<<Z_U <<endl;}
	output.close();

	//if(C_f&&C_t)*C_t=Z_U-*C_f;
}