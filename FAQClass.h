#pragma once

//http://qiita.com/Ugo-Nama/items/08c6a5f6a571335972d5

#include"Filer.h"

#include<functional>
#include<memory>
#include<vector>
#include<algorithm>
#include<random>
#include<iterator>
#include<iostream>
#include<map>
#include<utility>
#include<Randomer.h>
#include<DataBase.h>
#include<queue>
#include<tiny_dnn/tiny_dnn.h>
using namespace std;


//Q学習を行うクラス
//
//QLClass(const double &lr, const double &r, const double &e)
//lr=学習率
//r=割引率
//e=ランダムに手を打つ確率
template<typename S, typename A>
class FAQClass:public Randomer
{
public:

	using SA = std::pair<S, A>;
	using Q = double;

	using HashS = function<size_t(const S &s)>;
	using HashA = function<size_t(const A &a)>;

	struct eqstr
	{
		bool operator()(const SA &left, const SA &right) const
		{
			return left == right;
		}
	};

	using QTable = MLP;

	using SAQ = pair<SA, double>;

	using FuncR = std::function<Q(const S &s)>;
	using FuncT = std::function<S(const S &s, const A &a)>;
	using FuncAs = std::function<vector<A>(const S &s)>;

	using SAToInput = std::function<Eigen::VectorXd(const S &s, const A &a)>;

#ifdef UNICODE
	using FuncLoad = std::function<vector<SAQ>(const vector<vector<string>>&)>;
	using FuncWrite = function<void(const QTable&, vector<vector<string>>*)>;
#else
	using FuncLoad = std::function<vector<SAQ>(const vector<vector<wstring>>&)>;
	using FuncWrite = function<void(const QTable&, vector<vector<wstring>>*)>;
#endif // !UNICODE
private:
	double e;
	double discount_rate;
	QTable q_func;

	FuncT T;
	FuncR R;
	FuncAs As;

	FuncLoad load;
	FuncWrite write;

	SAToInput sa_to_input;
	A RandAction(const S &s)
	{
		auto pos_a = As(s);

		shuffle(begin(pos_a), end(pos_a), this->mt);

		auto ret = pos_a.front();
		auto _size = size(pos_a);
		return ret;
	}

public:
	FAQClass(MLP::Params &params)
		:q_func(params),discount_rate(0.5),e(0.5) {}

	void QUpDate(const S &s, const A &a)
	{
		auto s2 = this->T(s, a);
		auto a2 = BestAction(s2);
		auto r = this->R(s2);
		auto input = this->sa_to_input(s2, a2);
		double maxE = this->q_func.Forward(input)(0);

		MLP::Target target = MLP::Target::Zero(1);
		target(0) = r + this->discount_rate*maxE;

		MLP::DataList data_list{ make_pair(input, target) };
		this->q_func.Learn(data_list);
	}

	A Learn(const S &s)
	{
		uniform_real_distribution<> prop;

		A a;

		if (prop(this->mt) < this->e)
		{
			a = this->RandAction(s);
		}
		else
		{
			a = this->BestAction(s);
		}
		this->QUpDate(s, a);

		return a;
	}

	A BestAction(const S &s)
	{
		auto pos_a = As(s);

		vector<double> outputs;
		outputs.reserve(size(pos_a));

		for (auto &a : pos_a)
		{
			auto input = this->sa_to_input(s, a);
			auto output=this->q_func.Forward(input)(0);

			outputs.emplace_back(output);
		}

		auto itr = std::max_element(begin(outputs), end(outputs));
		auto index = std::distance(begin(outputs), itr);

		auto best_a = pos_a[index];

		return best_a;
	}

	void Disp()
	{
		this->q_func.Disp();
	}

	void SetFunc
	(
		const FuncR &func_r,
		const FuncT &func_t,
		const FuncAs &func_as,
		const FuncLoad &func_load,
		const FuncWrite &func_write,
		const SAToInput &sa_to_input
	)
	{
		this->R = func_r;
		this->T = func_t;
		this->As = func_as;
		this->load = func_load;
		this->write = func_write;
		this->sa_to_input = sa_to_input;
	}

};