from fastapi import HTTPException


def not_implemented(stage: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=501,
        detail={
            "code": "not_implemented",
            "stage": stage,
            "message": message,
        },
    )


def api_error(status: int, code: str, stage: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"code": code, "stage": stage, "message": message},
    )
