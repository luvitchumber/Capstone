from data_processing import *
from lstm_model import LSTMModel
from dict_bin_search import search
import pandas as pd
import os.path
import menu
import re
import sys

def get_dataset(sel_intersection, intersections):
	intersection_file = search(intersections, 'name', sel_intersection)
	intersection_file = intersections[intersection_file]['dataset']
	df = pd.read_csv(intersection_file.strip(' '))
	return df


def main():
	"""Program execution starts here."""
	intersections = read_intersections("intersections.data")
	intersection_list = []
	
	for i in range(len(intersections)):
		intersection_list.append(intersections[i]['name'].strip(' '))
		
	#sel_intersection = intersections[0]['name']
	sel_intersection = "4589"

	df = get_dataset(sel_intersection, intersections)
	x_train, x_test, y_train, y_test = preprocessing(df)

	model_file = search(intersections, 'name', sel_intersection)

	main_menu = [
		"CAPSTONE TRAFFIC PREDICTION",

		"Select Intersection",
		"List of Intersections",
		"Train Intersection",
		"Test Intersection (Accuracy)",
		"Route Check"
	]
	train_menu = [
		"Train Mode:",

		"Train from scratch w/ events",
		"Train from file w/ events",
		"Train from scratch",
		"Train from file",
	]
	route_check_menu = [
		"Train Mode:",

		"Train from scratch w/ events",
		"Train from file w/ events",
		"Train from scratch",
		"Train from file",
	]

	while True:
		print("Currently Selected Intersection:", sel_intersection)
		choice = menu.do_menu(main_menu)
		if choice is None:
			return  # Exit main() (and program).
		if choice == 1:
			# Select Intersection
			temp_menu = ["Please Select a New Intersection"]

			for line in intersections:
				option = line["name"] + ": " + line["street"]
				temp_menu.append(option)

			choice = menu.do_menu(temp_menu)
			if choice is not None:
				sel_intersection = intersections[choice - 1]['name']
				print(sel_intersection, "set as current intersection.")

				df = get_dataset(sel_intersection, intersections)
				x_train, x_test, y_train, y_test = preprocessing(df)

				model = LSTMModel(x_train.shape[1], y_train.shape[1])
				model_file = search(intersections, 'name', sel_intersection)
				if model_file is not None:
					model_file = intersections[model_file]['model']
					model.load_network(model_file)

		elif choice == 2:
			# List Intersections
			print_intersections(intersections)
		elif choice == 3:
			model = LSTMModel(x_train.shape[1], y_train.shape[1])
			# Train Intersections
			# TODO test each option
			choice = menu.do_menu(train_menu)
			if choice == 1:
				x_train, x_test, y_train, y_test = preprocessing(df, events=True)
				model = LSTMModel(x_train.shape[1], y_train.shape[1])

				intersection_idx = search(intersections, 'name', sel_intersection)
				model_file = "model/" + sel_intersection + ".hdf"
				if os.path.exists(model_file):
					os.remove(model_file)
				model.init_network(hidden_size=50)
			elif choice == 2:
				x_train, x_test, y_train, y_test = preprocessing(df, events=True)
				model = LSTMModel(x_train.shape[1], y_train.shape[1])

				intersection_idx = search(intersections, 'name', sel_intersection)
				model_file = "model/" + sel_intersection + ".hdf"
				if os.path.exists(model_file):
					model.load_network(model_file)
				else:
					print("Model does not exist, starting from scratch")
					model.init_network(hidden_size=50)
			elif choice == 3:
				x_train, x_test, y_train, y_test = preprocessing(df, events=False)
				model = LSTMModel(x_train.shape[1], y_train.shape[1])

				intersection_idx = search(intersections, 'name', sel_intersection)
				model_file = "model/" + sel_intersection + ".hdf"
				if os.path.exists(model_file):
					os.remove(model_file)
				model.init_network(hidden_size=50)
			elif choice == 4:
				x_train, x_test, y_train, y_test = preprocessing(df, events=False)
				model = LSTMModel(x_train.shape[1], y_train.shape[1])

				intersection_idx = search(intersections, 'name', sel_intersection)
				model_file = "model/" + sel_intersection + ".hdf"
				if os.path.exists(model_file):
					model.load_network(model_file)
				else:
					print("Model does not exist, starting from scratch")
					model.init_network(hidden_size=50)
			try:
				e = int(input("Enter Epochs to train for (Default=50): "))
				model.epochs = e
			except ValueError:
				print("Invalid number entered, using default value")
			model.train(x_train, y_train, model_file)

			intersections[intersection_idx]['model'] = model_file
			save_intersections(intersections, "intersections.data")

		elif choice == 4:
			# Test Intersections
			model_file = "model/" + sel_intersection + ".hdf"
			model = LSTMModel(x_train.shape[1], y_train.shape[1])
			if os.path.exists(model_file):
				model.load_network(model_file)
				test_output = model.get_accuracy(x_test, y_test)
				N, S, E, W = confusion_matrix(test_output, y_test, True)
			else:
				print("Please train intersection first")

		elif choice == 5:
			model = LSTMModel(x_train.shape[1], y_train.shape[1], intersection_list)
			# Route Check
			x_data = []
			day_week = [1,2,3,4,5,6,7]
			season=[1,2,3,4]
			time_str = ''
			flag = 0
			x_data = []
			peak = 0

			while flag == 0:
				num_inter = input("Please enter intersection number: ")
				if num_inter in intersection_list:
					flag = 1
					x_data.append(int(num_inter))
				else:
					print("Intersection not found!")
					
			flag = 0
			while flag == 0:
				week = [0,0,0,0,0,0,0]
				num_day = input("Please enter the day of the week:\nOptions:\n1:Sunday\n2:Monday\n3:Tuesday\n4:Wednesday\n5:Thursday\n6:Friay\n7:Saturday\n")
				num_day = int(num_day)
				if num_day in day_week:
					flag = 1
					week[num_day-1] = 1
					x_data = x_data + week
				else:
					print("Day of the week not found!")
			
			flag = 0
			while flag == 0:
				time_day = input("Please enter the time of the day (ex. 17:30): ")
				temp = time_day.split(':')
				if len(temp) == 2:
					hour = int(temp[0]) * 60
					if hour > 9 and hour < 17:
						peak = 1
					time = hour + int(temp[1])
					time_d = float(time) / float(1440)
					if time_d > 1.0:
						print("Please enter time in the proper format!")
					else:
						x_data.append(time_d)
						flag = 1
				else:
					print("Please enter time in the proper format!")
			
			flag = 0
			while flag == 0:
				seasons = [0,0,0,0]
				season_input = input("Please enter the season:\n1:Summer\n2:Fall\n3:Winter\n4:Spring\n")
				season_input = int(season_input)
				
				if season_input in season:
					flag = 1
					seasons[season_input-1] = 1
					x_data = x_data + seasons
				else:
					print("Season not found!")
			
			# add [0,0] for events
			x_data.append(peak)
			x_test = np.array([x_data] + [0,0])
			x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
			res = model.predict(x_test)
			print("Prediction: " + str(res))
			sleep(10)
			
			
main()
