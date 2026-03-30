import nonebot
from nonebot import logger
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_builtin_plugins("echo")
nonebot.load_from_toml("pyproject.toml")

logger.info(
    f"bot bootstrap complete adapter=OneBot V11 "
    f"host={getattr(driver.config, 'host', '127.0.0.1')} "
    f"port={getattr(driver.config, 'port', '8000')} "
    f"env={getattr(driver.config, 'environment', 'prod')}"
)

if __name__ == "__main__":
    logger.info("starting nonebot run loop")
    nonebot.run()
