from typing import Annotated
from pydantic import BaseModel, Field, ValidationError
import time
import asyncio

ShoeSize = Annotated[ int , Field( ge=34 , le=46 ) ]


class Shoe(BaseModel):
    code: str
    name: str
    colors: list[str]
    sizes: list[ShoeSize]
    price_bob: float = Field( ge=0 )
    in_stock: bool = True


def pydantic_drill() -> None:
    good = {
        "code" : "ZR-101",
        "name" : "Zapatilla Runner",
        "colors" : [ "rojo", "negro" ],
        "sizes" : [38, 40 , 42],
        "price_bob" : 59.99
    }

    bad_size = { **good , "sizes" : [48 , 50] }
    bad_price = { **good , "price_bob" : -10.0 }


    print(Shoe.model_validate(good))

    for label , data in [ ("bad_size", bad_size), ("bad_price", bad_price) ]:
        try:
            Shoe.model_validate(data)
        except ValidationError as e:
            print(f"{label.capitalize()}: {e}")



async def fake_call(i : int) -> int:
    await asyncio.sleep(1);

async def one_after_nother() -> None:
    for i in range(10):
        await fake_call(i)

async def all_at_once() -> None:
    await asyncio.gather(*[fake_call(i) for i in range(10) ])


async def three_at_a_time() -> None:
    sem = asyncio.Semaphore(3)
    start = time.perf_counter()

    async def limited_call(i: int) -> int:
        async with sem:
            print(f"  call {i} starts at {time.perf_counter() - start:.1f} s")
            return await fake_call(i)

        await asyncio.gather(*[limited_call(i) for i in range(10)])

async def asyn_drill() -> None:
    versions = [ 
        ("one after onother", one_after_nother),
        ("all at once", all_at_once),
        ("three at a time", three_at_a_time)
    ]

    for label , version in versions:
        start = time.perf_counter()
        await version()
        print(f"{label}: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    # pydantic_drill()
    asyncio.run(asyn_drill())