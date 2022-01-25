shop_prices = [30000, 2000, 1500000]
user_coupons = [20, 40]
#비싼거 순서대로 비싼할인율 적용하면 좋지 않을까? -> 정렬
#높은거 먼저보고 높은거 할인받고싶다.
def get_max_discounted_price(prices, coupons):
    coupons.sort(reverse=True) #그냥 sort는 오름차순/ reverse=True는 내림차순 정렬
    prices.sort(reverse=True)
    price_index = 0
    coupon_index = 0
    max_discounted_price = 0
    #쿠폰과 가격의 갯수가 다를 수 있기 때문에 for문대신 while문을 쓰자
    while price_index < len(prices) and coupon_index < len(coupons):
        #할인율이 퍼센트라서 100을 꼭 나눠야 하고 남은 것을 해야하기 때문에 100에다가 빼야한다.
        max_discounted_price += prices[price_index] * (100 - coupons[coupon_index]) / 100
        price_index += 1
        coupon_index += 1

    while price_index < len(prices):
        max_discounted_price += prices[price_index]
        price_index += 1

    return max_discounted_price


print(get_max_discounted_price(shop_prices, user_coupons))  # 926000 이 나와야 합니다.