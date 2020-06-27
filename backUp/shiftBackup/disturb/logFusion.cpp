#include <iostream>
#include <fstream>
#include <iomanip>

#define RIGHT_OFFSET 40

int		nonPrintableSize(std::string str) {
		int			pos = 0;
		int			i = 0;
		
		while ((pos = str.find("\033", pos)) != std::string::npos) {
			i++;
			pos++;
		}
		i /= 2;
		return (i * (5 + 4));
}



int		main(int	argc, char ** argv) {
	if (argc != 2) {
		std::cout << "usage: ./logFusion match_dir_path\nwith left.disturb.log et right.disturb.log in it" << std::endl;
		return 0;
	}
	std::ifstream			inLeft(argv[1] + std::string("/left.disturb.log")), inRight(argv[1] + std::string("/right.disturb.log"));
	std::ofstream			ofs(argv[1] + std::string("/fusion.disturb.log"));
	std::string				str[3];

	while (1) {
		getline(inLeft, str[0]);
		getline(inRight, str[1]);
//		std::cout << str[0].length() << std::endl;
//		std::cout << nonPrintableSize(str[0]) << std::endl;
		str[2] = std::string(RIGHT_OFFSET - (str[0].length() - nonPrintableSize(str[0])), ' ');
		ofs << str[0] << str[2] << str[1] << std::endl;
		if (!inLeft && !inRight)
			break;
	}

	return 0;
}
