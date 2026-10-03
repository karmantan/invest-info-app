def shadow_result(principal_eur, entry_price, current_price):
    if principal_eur<0 or entry_price<=0 or current_price<0: raise ValueError("invalid shadow position values")
    ending=principal_eur*current_price/entry_price
    return {"principal_eur":principal_eur,"ending_value_eur":ending,"result_eur":ending-principal_eur,"return":current_price/entry_price-1}

def compare_results(model_ending_eur, shadow_ending_eur, principal_eur):
    return {"model_profit_eur":model_ending_eur-principal_eur,"passive_profit_eur":shadow_ending_eur-principal_eur,"value_added_eur":model_ending_eur-shadow_ending_eur}

