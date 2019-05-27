import torch

class MCTSvisualiser:
    '''
    Class that aims to visualize MCTS. It generates .gv file that describe the tree. Then dot from Graphviz can
    be used to transform the .gv file into a pdf file with the tree grpahical representation on it.
    '''
    def __init__(self, env, indent = '    '):
        self.indent = indent
        self.env = env
        self.file_path = ''


    def _print_heading(self):
        '''
        Opens the .gv file and print the first line that declares the graph.
        '''
        self.file = open(self.file_path, 'w')
        self.file.write('digraph g{ \n')


    def _print_footing(self):
        '''
        Write the last line to end the graph declaration and closes the .gv file.
        Returns:

        '''
        self.file.write('}')
        self.file.close()


    def _get_label_attr(self, list_args, font_size=10):
        '''
        Convert a list of string arguments into the string understandable by graphviz.
        Args:
            list_args: list of string arguments
            font_size: font size

        Returns: the string understandable by graphviz

        '''
        res = '[label=<<FONT POINT-SIZE="{}">'.format(font_size)
        for idx,arg in enumerate(list_args):
            if idx != len(list_args) - 1:
                res += arg + ' <br/> '
            else:
                res += arg + ' '
        res += '</FONT>>]'
        return res

    def get_breadth_first_nodes(self, root_node):
        '''
        Performs a breadth first search inside the tree.

        Args:
            root_node: tree root node

        Returns:
            list of the tree nodes sorted by depths
        '''
        nodes = []
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node['childs']:
                stack.append(child)
        return nodes


    def print_mcts(self, root_node, file_path):
        '''
        Create a .gv file at file_path and put the tree representation in it.

        Args:
            root_node: root node of the tree
            file_path: path to save the .gv file
        '''
        self.file_path = file_path
        # open text file and print first line
        self._print_heading()
        # get nodes with a breadth first search
        nodes = self.get_breadth_first_nodes(root_node)
        # keep only visited nodes
        nodes = list(filter(lambda x: x['visit_count'] > 0, nodes))
        # Index nodes
        for idx, node in enumerate(nodes):
            node['index'] = idx

        # gather nodes per depth
        max_depth = nodes[-1]['depth']
        nodes_per_depth = {}
        for d in range(0, max_depth + 1):
            nodes_per_depth[d] = list(filter(lambda x: x['depth'] == d, nodes))

        # print tree layer per layer
        for d in range(0, max_depth + 1):
            # gather nodes for this layer
            nodes_this_depth = nodes_per_depth[d]
            nodes_idx_same_rank = [x['index'] for x in nodes_this_depth]
            # print nodes
            for node in nodes_this_depth:
                self._print_mcts_node(node)
            # print ranks
            self._print_same_rank(nodes_idx_same_rank)

            # print edges
            for node in nodes_this_depth:
                for child in node['childs']:
                    if child['visit_count'] > 0:
                        if child['selected']:
                            color = 'red'
                        else:
                            color = None
                        self._print_mcts_edge(node, child, color)

        # print last line and close text file
        self._print_footing()


    def _print_node(self, node_idx, list_args, font_size=10):
        self.file.write(self.indent + str(int(node_idx)) + ' ' + self._get_label_attr(list_args, font_size, ) + '\n')

    def _print_mcts_node(self, node):
        prior = node['prior']
        if prior is not None:
            prior = '%.2f'%(prior)
        values = torch.FloatTensor(node['total_action_value'])
        softmax = torch.exp(1.0 * values)
        softmax = softmax / softmax.sum()
        qvalue = float(torch.dot(softmax, values))
        qvalue = '%.2f'%(qvalue)


        list_args=['prog : {}'.format(self.env.get_program_from_index(node['program_index'])),
                   'env state: {}'.format(self.env.get_state_str(node["env_state"])),
                   'prior : {},  qvalue : {}'.format(prior, qvalue),
                   'depth : {}'.format(node['depth'])]

        self._print_node(node['index'], list_args)

    def _print_mcts_edge(self, parent, child, color=None):
        self._print_edge(parent['index'], child['index'], label=self.env.get_program_from_index(child['program_from_parent_index']), color=color)

    def _print_same_rank(self, list_nodes):
        res = '{rank=same;'
        for node in list_nodes:
            res += ' ' + str(int(node)) + ';'
        res += '}'
        self.file.write(self.indent + res + '\n')


    def _print_edge(self, node1_idx, node2_idx, label='', color=None, font_size=10):
        res = '{} -> {} '.format(str(int(node1_idx)), str(int(node2_idx)))
        res += '[ '
        if color is not None:
            res += 'color={}, '.format(color)
        res += 'label=<<FONT POINT-SIZE="{}">{}</FONT>>'.format(font_size, label)
        res += '];'
        self.file.write(self.indent + res + '\n')
