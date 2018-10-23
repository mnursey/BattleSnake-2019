import os
import json

export_loc = 'formatted_data'

def simple_setup_test_one(data_loc):
    global json

    if not os.path.isdir(export_loc):
        os.mkdir(export_loc)

    test_name = '1vs1_10by10_1f_2018snake'

    print('Data Formatter: formatting data for test ' + test_name)

    file_names = os.listdir(data_loc)

    output = open(export_loc + '/' + test_name + '.csv', 'w')

    output.write('move, food_x, food_y, mhealth, ehealth, mb0_x, mb0_y, mb1_x, mb1_y, mb2_x, mb2_y, mb3_x, mb3_y, mb4_x, mb4_y, mb5_x, mb5_y, mb6_x, mb6_y, mb7_x, mb7_y, mb8_x, mb8_y, mb9_x, mb9_y, eb0_x, eb0_y, eb1_x, eb1_y, eb2_x, eb2_y, eb3_x, eb3_y, eb4_x, eb4_y, eb5_x, eb5_y, eb6_x, eb6_y, eb7_x, eb7_y, eb8_x, eb8_y, eb9_x, eb9_y\n')

    for file_name in file_names:
        if(file_name.endswith('.json')):
            file = open(data_loc + '/' + file_name, 'r')
            raw_json = file.read()
            log_obj = json.loads(raw_json)

            for turn in log_obj:
                skip = False
                move = turn['move']

                if move is 'None':
                    continue

                apple_x = turn['input']['board']['food'][0]['x'] + 1
                apple_y = turn['input']['board']['food'][0]['y'] + 1

                my_snake_health = turn['input']['you']['health']
                enemy_snake_health = 0

                my_snake_body = []
                enemy_snake_body = []

                my_id = turn['input']['you']['id']

                if len(turn['input']['you']['body']) > 10:
                    skip = True

                for i in range(10):
                    if(i < len(turn['input']['you']['body'])):
                        my_snake_body.append(turn['input']['you']['body'][i]['x'] + 1)
                        my_snake_body.append(turn['input']['you']['body'][i]['y'] + 1)
                    else:
                        my_snake_body.append(0)
                        my_snake_body.append(0)

                for snake in turn['input']['board']['snakes']:
                    if snake['id'] is not my_id:
                        enemy_snake_health = snake['health']

                        if len(snake['body']) > 10:
                            skip = True

                        for i in range(10):
                            if(i < len(snake['body'])):
                                enemy_snake_body.append(snake['body'][i]['x'] + 1)
                                enemy_snake_body.append(snake['body'][i]['y'] + 1)
                            else:
                                enemy_snake_body.append(0)
                                enemy_snake_body.append(0)
                        break
            
                if skip:
                    continue

                out = move + ', ' + str(apple_x) + ', ' + str(apple_y) + ', ' + str(my_snake_health) + ', ' + str(enemy_snake_health)

                for item in my_snake_body:
                    out += ', ' + str(item)

                for item in enemy_snake_body:
                    out += ', ' + str(item)

                output.write(out + '\n')
            file.close()
    output.close()

    print('Data Formatter: finished formatting data')

    return

if __name__ == '__main__':
   simple_setup_test_one('./log_data/1v1_10by10_1f_2018snake')