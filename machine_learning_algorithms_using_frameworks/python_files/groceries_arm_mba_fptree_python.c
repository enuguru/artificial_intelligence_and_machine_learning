#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define FALSE 0
#define TRUE 1


float minSupport = 0.05;
int SUPPORT = 0;
char *nullChar = "\0";

typedef struct nodes{	//nodes of fpTree
	int count;	//count of item instances
	int item;	//item name
	struct nodes *parent;	//parent address
	struct nodes *sibling;	//pointer to next instance of the same item
	struct nodes *child;
	struct nodes *level;
}node;

typedef struct itemsTables{
	int count;	//count of item instances
	char *item;	//item name
	node *reference;	//address of the instance in fpTree
	node *last_sibling;
}itemsTable;

typedef struct miningTables{
	int count;
	char item;
	node *reference;
	node *last_sibling;
}miningTable;

struct Ordering{
		int array[100];
		int len;
		int count;
};
typedef struct Ordering order;

// initialization for each node in fp-tree, items_table, etc.

void initialize_node(node* temp){
    temp->count = 0;
	temp->item = -1;
    temp->parent = temp->sibling = temp->child = temp->level = NULL;
}
void initialize_items_table(itemsTable* temp){
    temp->count = 0;
    temp->item = nullChar;
    temp->reference = NULL;
	temp->last_sibling = NULL;
}
void initialize_mining_table(miningTable* temp){
    temp->count = 0;
    temp->item = -1;
    temp->reference = NULL;
	temp->last_sibling = NULL;
}
void initialize_order(order * temp){
	temp->len = 0;
	temp->count = 0;
}
int find_index(int item, miningTable miningCounter[], int miningCount){
	int i = -1;
	for(i = 0; i <= miningCount; ++i){
		if(miningCounter[i].item == item)
			break;
	}
	return i;
}

void sortDesc(itemsTable itemsCounter[10000], int last_index)
{
	/*
		Sorts the entries in itemsCounter in descending order.
	*/
    int i, j;
    itemsTable temp;
    
    for (i = 0; i < last_index - 1; i++)
    {
        for (j = 0; j < (last_index - 1-i); j++)
        {
            if (itemsCounter[j].count < itemsCounter[j + 1].count)
            {
                temp = itemsCounter[j];
                itemsCounter[j] = itemsCounter[j + 1];
                itemsCounter[j + 1] = temp;
            } 
        }
    }
}

void supportFilter(itemsTable itemsCounter[10000], int* item_count)
{
	/*
		Discards items from itemsCounter having frquency less than the support count.
	*/
	int last_index = *item_count;
	int supportCount = SUPPORT;
	
	int i=0,j;
	while(i<=last_index){
		if(itemsCounter[i].count<supportCount){
			for(j=i;j<last_index;j++){
				itemsCounter[j] = itemsCounter[j+1];
			}
			last_index--;
		}
		else{
			i++;		
		}	
	}
	*item_count = last_index;

}

void displayItems(itemsTable itemsCounter[], int item_count)
{
	/*
		display items in items counter along with frequenies.
	*/
	for (int j=0;j <= item_count;j++){
   		printf("**** %d  - %s -  %d *******\n", j, itemsCounter[j].item, itemsCounter[j].count);
    }
}
void orderTransaction(char* transaction_data, itemsTable itemsCounter[], order table[], int order_row, int item_count ){
	
	/*
		Orders transaction according to decreasing order of frequencies.
	*/
	char* data_item = (char*)malloc(1000*sizeof(char));

	int k = 0;
	int trans_iter = 0;
	int found;
	initialize_order(&table[order_row]);

	for(int i = 0; i <= item_count; ++i )
	{
		trans_iter = 0;
		found = FALSE;
		while(transaction_data[trans_iter] != '\0' && found == FALSE){
			
			if(transaction_data[trans_iter] != ',')
				data_item[k++] = transaction_data[trans_iter];
			else{
				data_item[k] = '\0';
				if(strcmp(data_item, itemsCounter[i].item) == 0){
					table[order_row].array[table[order_row].len++] = i;
					found = TRUE;					
				}
				free(data_item);
				data_item = NULL;
				data_item = (char*)malloc(1000*sizeof(char));
				k = 0;
			}
			trans_iter++;
		}
	}
	
}

int orderTable(itemsTable itemsCounter[10000], order table[10000], int item_count)
{
	/*
		Orders table of transactions using orderTransaction function
	*/
	
	//displayItems(itemsCounter, item_count);
	
	FILE* stream = fopen("groceries_subset.csv", "r");
	if(stream == NULL){
    	perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
	
	char ch;	
	char*  transaction_data = (char*)malloc(1000*sizeof(char));
	strcpy(transaction_data, "\0");
	char *data_item = (char*)malloc(1000*sizeof(char));
	
	int i=0, j=0, items_i=0, last_index=-1, flag, seen1 = 0, seen2 = 0, totalTrans = 0, order_row = -1;
	char prev_ch = '1';	
    while((ch = fgetc(stream)) != EOF){ 

    	if(isalpha(ch) || ch==' ' ||ch == '/' || ch == '-' || ch == '.'){
			
    		if(seen1!=0){
    			data_item[i]=ch;
    			i++;
    			seen2=1;
    		}
    	}
    	
    	else if(ch==','|| ch=='\n'){
			if(isdigit(prev_ch)){
				totalTrans++;
				order_row++;
				if(totalTrans > 1)
				{
					orderTransaction( transaction_data, itemsCounter, table, order_row - 1, item_count);
					free(transaction_data);
					transaction_data = NULL;
					transaction_data = (char*)malloc(10000*sizeof(char));
					strcpy(transaction_data, "\0");
				}	
		}
    		if(seen1 == 0){
    			seen1 = 1;
    		}
    		else if(seen2==1){
				seen2 = 0;
				data_item[i]=',';
				data_item[i+1] = '\0';
				strcat(transaction_data, data_item);
		
				free(data_item);
				data_item = NULL;
				data_item = (char*)malloc(100*sizeof(char));
				i=0;
			}
		}
		prev_ch = ch;
    }
	
	orderTransaction(transaction_data, itemsCounter, table, order_row, item_count);
	free(transaction_data);
	transaction_data = NULL;
    
	fclose(stream); 	
	return totalTrans;
}

void constructBaseFPtree(itemsTable itemsCounter[], order table[], int row_count)
{
	/*
		constructs initial FP tree based on transactions from the file.
	*/
	    node *parent, *root, *current, *level_traverse;
        //phase 1
        root = (node*)malloc(sizeof(node));
        initialize_node(root);
        for(int i = 0; i < row_count; ++i){
            current = root;
            for(int j = 0; j < table[i].len; ++j){
                int found = FALSE;
                parent = current;
                current = parent->child;
                if(current == NULL && found == FALSE){                    
                    current = (node*)malloc(sizeof(node));
                    initialize_node(current);
                    current->item = table[i].array[j];
                    current->count = 1;
                    current->parent = parent;
					parent->child = current;
                }
                else{
					node ** prev_level_pointer;
                    level_traverse = current;
                    while(level_traverse != NULL && found == FALSE){
                        if(level_traverse->item == table[i].array[j]){
                            found = TRUE;
                            level_traverse->count++;
                            break;
                        }
						prev_level_pointer = &(level_traverse->level);
                        level_traverse = level_traverse->level;
                    }
                    if(level_traverse == NULL){
                        level_traverse = (node*)malloc(sizeof(node));
                        initialize_node(level_traverse);
                        level_traverse->item = table[i].array[j];
                        level_traverse->count = 1;
                        level_traverse->parent = parent;
						*prev_level_pointer = level_traverse;
                    }                
                    current = level_traverse;
                }
                //setting sibling links
				if(found == FALSE){
					int l = current->item;
					if(itemsCounter[l].reference == NULL) itemsCounter[l].reference = current;
					node** prev_sib_link = &(itemsCounter[l].last_sibling->sibling);
					itemsCounter[l].last_sibling = current;
					if(itemsCounter[l].last_sibling != itemsCounter[l].reference)
						*prev_sib_link = current;
				}                
			}
		}
}
void constructFPtree(miningTable miningCounter[], int miningCount, order table[], int row_count)
{
	/*
		Construct FP tree.
	*/
	    node *parent, *root, *current, *level_traverse;
        //phase 1
        root = (node*)malloc(sizeof(node));
        initialize_node(root);
        for(int i = 0; i < row_count; ++i){
            current = root;
            for(int j = 0; j < table[i].len; ++j){
                int found = FALSE;
                parent = current;
                current = parent->child;
                if(current == NULL && found == FALSE){                    
                    current = (node*)malloc(sizeof(node));
                    initialize_node(current);
                    current->item = table[i].array[j];
                    current->count = table[i].count;
                    current->parent = parent;
					parent->child = current;
                }
                else{
					node ** prev_level_pointer;
                    level_traverse = current;
                    while(level_traverse != NULL && found == FALSE){
                        if(level_traverse->item == table[i].array[j]){
                            found = TRUE;
                            level_traverse->count += table[i].count;
                            break;
                        }
						prev_level_pointer = &(level_traverse->level);
                        level_traverse = level_traverse->level;
                    }
                    if(level_traverse == NULL){
                        level_traverse = (node*)malloc(sizeof(node));
                        initialize_node(level_traverse);
                        level_traverse->item = table[i].array[j];
                        level_traverse->count = table[i].count;
                        level_traverse->parent = parent;
						*prev_level_pointer = level_traverse;
                    }                
                    current = level_traverse;
                }
                //setting sibling links
				if(found == FALSE){
					int current_item = current->item;
					int l = find_index(current_item, miningCounter, miningCount);
					
					if(miningCounter[l].reference == NULL) miningCounter[l].reference = current;
					node** prev_sib_link = &(miningCounter[l].last_sibling->sibling);
					miningCounter[l].last_sibling = current;
					if(miningCounter[l].last_sibling != miningCounter[l].reference)
						*prev_sib_link = current;
				}                
			}
		}
}
void print_tree(itemsTable itemsCounter[], int item_count){
	/*
		prints FP tree
	*/
    node* current_node,* current_sibling;
    for(int i = 0; i <= item_count; ++i){
        printf("item: %d count: %d item_sets: ", i, itemsCounter[i].count);
        current_sibling = itemsCounter[i].reference;
        
		int k = 0;
        while(current_sibling != NULL){
			k++;
            current_node = current_sibling;
            printf("\n");
            while(current_node->parent != NULL){
                printf("%d ", current_node->item);
                current_node = current_node->parent;
            }
            current_sibling = current_sibling->sibling;
        }
		printf("%d\n",k);
    }
	printf("\n");
}

void copyToMiningTable(itemsTable itemsCounter[], miningTable miningCounter[], int item_count){
	/*
		Copies contents of itemsCounter to mining Counter except the item string.
	*/
	for(int i = 0; i <= item_count; ++i){
		initialize_mining_table(&miningCounter[i]);
		miningCounter[i].item = i;
		miningCounter[i].count = itemsCounter[i].count;
		miningCounter[i].last_sibling = itemsCounter[i].last_sibling;
		miningCounter[i].reference = itemsCounter[i].reference;
	}
}
void mineFPtree(itemsTable itemsCounter[],miningTable old_miningCounter[],int old_miningCount,order old_table[],int old_row_count,int prefix[]){
	/*
		Recursive function for mining FP tree.
	*/
	if(old_miningCount < 0) {
		printf("\n--------------------- end ---------------------\n");
		return;

	}
	char prefix_string[1000] = "\0";
	int last_index = 0;
	
	while(prefix[last_index] != -1){
		strcat(prefix_string, itemsCounter[ prefix[last_index] ].item);
		strcat(prefix_string, ",");
		last_index++;
	}
	

	for(int i = 0; i <= old_miningCount; ++i){
		
		order table[1000];
		miningTable miningCounter[1000];
		int new_prefix[1000];
		int miningCount = -1;
		int row_count = 0;

		int item = old_miningCounter[i].item;
		int item_total_count = old_miningCounter[i].count;

		printf("main item: %d\n", item);
		printf("prefix: \n");
		for(int j = 0; j < last_index; ++j)
		{	
			new_prefix[j] = prefix[j];
			printf("%d ", new_prefix[j]);
		}
		new_prefix[last_index] = item;
		printf("%d\n", item);
		new_prefix[last_index + 1] = -1;

		printf("transaction:: {%s%s}:%d\n", prefix_string, itemsCounter[item].item, old_miningCounter[i].count);

		node* current_node,*current_sibling;
		
			current_sibling = old_miningCounter[i].reference;
			
			int k = 0;
			while(current_sibling != NULL){
				initialize_order(&table[k]);
				int sibling_count = current_sibling->count;
				table[k].count = sibling_count;

				current_node = current_sibling->parent;
				while(current_node->parent != NULL){
					int current_item = current_node->item;
					int found = FALSE;
					table[k].array[ table[k].len++ ] = current_item;

					for(int j = 0; j <= miningCount && found == FALSE; ++j){
						if(miningCounter[j].item == current_item)
						{
							found = TRUE;
							miningCounter[j].count += sibling_count;
						}
					}

					if(found == FALSE){
						miningCount++;
						initialize_mining_table(&miningCounter[ miningCount ]);
						miningCounter[ miningCount ].item = current_item;
						miningCounter[ miningCount ].count = sibling_count;
					}
					current_node = current_node->parent;
				}
				current_sibling = current_sibling->sibling;
				k++;
			}
			row_count = k;
			int supportCount = SUPPORT;
			for(int j = 0; j < miningCount; ++j){
				for(int k = j+1; k <= miningCount; ++k){
					if(miningCounter[j].count < miningCounter[k].count){
						int temp = miningCounter[j].item;
						miningCounter[j].item = miningCounter[k].item;
						miningCounter[k].item = temp;
						temp = miningCounter[j].count;
						miningCounter[j].count = miningCounter[k].count;
						miningCounter[k].count = temp;
					}
				}
			}
			for(int j = 0; j < miningCount; ++j){
				if(miningCounter[j].count < supportCount)
				{
					miningCount = j - 1;
					break;
				}
			}
			
			if(miningCount >= 0 && miningCounter[miningCount].count < supportCount)
				miningCount--;
			
			for(int k = 0; k < row_count; ++k){
				int found = FALSE;
				int current_index = 0;
				for(int j = 0; j <= miningCount; ++j){
					for(int l = current_index; l < table[k].len; ++l){
						if(table[k].array[l] == miningCounter[j].item ){
							if(l != current_index){
								int temp = table[k].array[l];
								table[k].array[l] = table[k].array[current_index];
								table[k].array[current_index] = temp;
							}
							current_index++;
						}
					}
				}
				table[k].len = current_index;
			}
			constructFPtree(miningCounter, miningCount, table, row_count);
			
			mineFPtree(itemsCounter,miningCounter,miningCount,table,row_count,new_prefix);
			
	}//conditional mining
}

int main(){
	
	FILE* stream = fopen("groceries_subset.csv", "r");
	if(stream == NULL){
    	perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
	
	char ch;	
	char *data_item = (char*)malloc(1000*sizeof(char));
	itemsTable itemsCounter[10000];
	miningTable miningCounter[10000];
	order table[10000];

	initialize_items_table(&itemsCounter[0]);
	
	int i=0, j=0, items_i=0, last_index=-1, flag, seen1 = 0, seen2 = 0, totalTrans = 0;
	char prev_ch = '1';	
    while((ch = fgetc(stream)) != EOF){ 

    	if(isalpha(ch) || ch==' ' ||ch == '/' || ch == '-' || ch == '.'){
			
    		if(seen1!=0){
    			data_item[i]=ch;
    			i++;
    			seen2=1;
    		}
    	}
    	
    	else if(ch==','|| ch=='\n'){
			if(isdigit(prev_ch)) totalTrans++;
    		if(seen1 == 0){
    			seen1 = 1;
    		}
    		else if(seen2==1){
				seen2 = 0;
				data_item[i]='\0';
				// 
				flag=0;
				for(j=0;j<=last_index && flag == 0;j++){
					if(strcmp(itemsCounter[j].item,data_item) == 0){
						itemsCounter[j].count++;
						flag = 1;
						// 
						break;
					}
				}	
				if(flag==0){
					last_index++;
					initialize_items_table(&itemsCounter[last_index]);
					itemsCounter[last_index].item = data_item;
					itemsCounter[last_index].count = 1;	
				}
				// 
				data_item = NULL;
				data_item = (char*)malloc(100*sizeof(char));
				i=0;
				if(ch=='\n'){
					seen1 = 0;
				}
			}
		}
		prev_ch = ch;
    }
	
	fclose(stream); 	

    sortDesc(itemsCounter, last_index+1);
    
	SUPPORT = ceil(totalTrans*minSupport);
    supportFilter(itemsCounter, &last_index);

	
    int row_count = orderTable(itemsCounter,table, last_index);
	constructBaseFPtree(itemsCounter, table, row_count);
	copyToMiningTable(itemsCounter, miningCounter, last_index);
	
	int prefix[1000];
	prefix[0] = -1;
	mineFPtree(itemsCounter, miningCounter, last_index, table, row_count, prefix);
	return 0;
}
