__author__ = 'David T. Pocock'


class Parser:

    def __init__(self, path):
        self.path = path

    def parse_data(self):
        with open(self.path, newline='') as file:
            num_of_levels = int(file.readline())

            concat_reqs = []
            for level in range(0, num_of_levels):
                next(file)
                concat_reqs.append(file.readline().split())
            req_costs = [int(i) for sublist in concat_reqs for i in sublist]
            requirements = dict(enumerate([[] for sublist in concat_reqs for i in sublist]))

            num_of_deps = int(file.readline())
            for dep in range(0, num_of_deps):
                next(file)

            num_of_custs = int(file.readline())
            customers = []
            cust_weights = []
            for i in range(0, num_of_custs):
                customer = file.readline().split()
                del customer[1]
                cust_weights.append(int(customer.pop(0)))

                req_list = list(map(int, customer))
                customers.append(req_list)
                for num_of_req in customer:
                    requirements[int(num_of_req) - 1].append(i + 1)

            total_cust_weights = sum(cust_weights)
            norm_cust_weights = list(map(lambda x: x / total_cust_weights, cust_weights))
            norm_customers = list(zip(norm_cust_weights, customers))

            print("Requirements:", len(requirements))
            print("Customers:", len(customers))
            print("Total Customer Weights:", total_cust_weights)

        return requirements, norm_customers, req_costs
