// FAQ.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

#include"FAQClass.h"
#include<conio.h>

using namespace std;

int main()
{

	using S = int;
	using A = int;
	using FAQ = FAQClass<S, A>;
	using Q = FAQ::Q;
	using SA = FAQ::SA;
	using SAQ = FAQ::SAQ;

	enum
	{
		UP,
		LEFT,
		DOWN,
		RIGHT,
	};

	const vector<S> s_list =
	{
		0,1,2,
		3,4,5,6,7,
		8,9,10,
	};

	const vector<A> a_list =
	{
		UP,LEFT,DOWN,RIGHT,
	};

	FAQ::FuncR r = [](const S &s)
	{
		Q ret;

		if (s == 3)
		{
			ret = -10.;
		}
		else if (s == 7)
		{
			ret = 10.;
		}
		else
		{
			ret = -1.;
		}

		return ret;
	};

	FAQ::FuncT t = [&s_list](const S &s, const A &a)
	{
		int d[] = { -4,-1,+4,+1, };
		S ret = s_list[s] + d[a];

		auto not_up = ret < 0;
		auto not_down = ret >= 11;
		auto not_left = (d[a] == LEFT) && (s_list[s] == 0 || s_list[s] == 8);
		auto not_right = (d[a] == RIGHT) && (s_list[s] == 2 || s_list[s] == 10);

		if (s_list[s] == 3 || s_list[s] == 7)
		{
			ret = 5;
		}
		else if (not_up || not_down || not_left || not_right)
		{
			ret -= d[a];
		}

		return ret;

	};

	FAQ::FuncAs as = [&a_list](const S &s)
	{
		vector<A> ret(begin(a_list), end(a_list));
		return ret;
	};

	FAQ::FuncLoad load = [](const vector<vector<string>> &str)
	{
		vector<SAQ> ret;
		return ret;
	};

	FAQ::FuncWrite write = [](const FAQ::QTable &saq, vector<vector<string>> *str)
	{
	};

	FAQ::SAToInput sa_to_input = [](const S &s, const A &a)
	{
		using namespace Eigen;

		vector<double> data{ 1.*s,1.*a };
		MLP::Input ret = Map<MLP::Input>(data.data(), 2);

		return ret;
	};

	MLP::Params params;

	auto &layer_info = params.first;
	auto &loss = params.second;

	layer_info.push_back(make_pair(2, move(make_unique<ReLu>())));
	layer_info.push_back(make_pair(3, move(make_unique<ReLu>())));
	layer_info.push_back(make_pair(1, move(make_unique<ReLu>())));

	loss = make_unique<MSE>();

	FAQ obj(params);
	obj.SetFunc(r, t, as, load, write,sa_to_input);

	S s = 5;
	A a;

	for (int i = 0; i < 1000; ++i)
	{
		a = obj.Learn(s);
		s = t(s, a);
	}

	obj.Disp();
	_getch();
	return 0;
}

